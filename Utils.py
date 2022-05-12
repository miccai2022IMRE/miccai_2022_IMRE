from base64 import encode
import torch
import os
import sys
import scipy
from scipy import io
import math
import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Constants import device, train_h_folder, val_h_folder
from Models.PHER import PHER, Var_Encoder
from torch.utils.data import DataLoader
#from Dataset import CustData
import Dataset as Ds
import pickle
import re

from Constants import H,W, latent_dim


class UtilityFunctions ():

    def __init__(self):
        pass

    @staticmethod
    def find_min_max(h_path):

        files = os.listdir(h_path)
        tot_data = len(files)
        xmin = 1000
        xmax = -1000
        for i, file in enumerate(files):

            path = os.path.join(h_path, file)
            matFile = io.loadmat(path)
            x = matFile['H']
            if x.min()< xmin:
                xmin = x.min()
            if x.max()> xmax:
                xmax = x.max()

        return xmin, xmax


    @staticmethod
    def temp_cc(bsl_x, x):
        
        cc = 0
        cc_2 = 0
        wlen = x.shape[0]
        for s in range(wlen):
            corcof = np.corrcoef(bsl_x[s,:], x[s,:])
            cc_2 += corcof[0,1]

        return float(cc_2/wlen)


    @staticmethod
    def rmse_er(bsl_x, x):
        
        bsl_x = np.squeeze(np.asarray(bsl_x.flatten()))
        x = np.squeeze(np.asarray(x.flatten()))
        mse = skm.mean_squared_error(bsl_x, x)
        rmse = math.sqrt(mse)
        return rmse

    
    @staticmethod
    def space_cc(bsl_x, x):
        cc = 0
        cc_2_s = 0
        tlen = x.shape[1]
        for s in range(tlen):
            corcof_s = np.corrcoef(bsl_x[:,s].reshape(1,-1), x[:,s].reshape(1,-1))
            cc_2_s += corcof_s[0,1]

        return float(cc_2_s/tlen)


    @staticmethod
    def plot_losses(train_a, test_a, loss_type, num_epochs):
        
        """
            Plot epoch against train loss and test loss 
        """
        # plot of the train/validation error against num_epochs
        fig, ax1 = plt.subplots(figsize=(6, 5))
        ax1.set_xticks(np.arange(0 + 1, num_epochs + 1, step=10))
        ax1.set_xlabel('epochs')
        ax1.plot(train_a, color='green', ls='-', label='train {} loss'.format(loss_type))
        ax1.plot(test_a, color='red', ls='-', label='test {} loss'.format(loss_type))
        h1, l1 = ax1.get_legend_handles_labels()
        ax1.legend(h1, l1, fontsize='14', frameon=False)
        ax1.grid(linestyle='--')
        plt.tight_layout()
        fig.savefig('img/{}_{}.png'.format(loss_type, latent_dim), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()


    @staticmethod
    def loss_function(h_rot, recon_hrot, mu, logvar, beta):

        N, h, w = h_rot.shape
        mse_H = F.mse_loss(h_rot, recon_hrot.view(N, h, w), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total = mse_H + beta * KLD

        return mse_H , KLD, total


    @staticmethod
    def inline_print(s):
        sys.stdout.write(s + '\r')
        sys.stdout.flush()


    @staticmethod
    def save_recons (model_path, out_folder):

        model = PHER()
        model.to(device)

        #model_path = "Model_Weights/modl_no_elu_last_detenc_500_16_epoch1950_latent0.001_lr0.0001"

        model.load_state_dict(torch.load(model_path))
        model.eval()


        xmin, xmax = UtilityFunctions.find_min_max (train_h_folder)

        val_dataset = Ds.CustData(val_h_folder, xmin, xmax)    
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)
        
        no_of_save = 10
        idx = 0

        for hbsl, hrot, hboth in val_loader:
            hbsl = hbsl.to(device)
            hrot = hrot.to(device)
            hboth = hboth.to(device)

            mu, logvar, z, recon_h = model(hbsl, hrot, hboth)

            original_h = hrot[0].detach().cpu().numpy()
            recon_h = recon_h[0].detach().cpu().numpy()

            scipy.io.savemat (os.path.join (out_folder, "Original_"+str(idx)+".mat"),\
                 mdict={'H_Org': original_h})

            scipy.io.savemat (os.path.join(out_folder,"Recon_"+str(idx)+".mat"), \
                 mdict={'H_Recon': recon_h})

            idx += 1
            if idx == no_of_save:
                break


    #@staticmethod
    def reconstruction_zchange(model_path, out_folder):


        """
        Generate new samples by changing Z and keep H_i fixed
            - generate a z_det for H_i
            - repeat z_det for number of samples to create
            - sample no_of_samples to generate, from normal dist
        """

        model = PHER()
        model.to(device)

        model.load_state_dict(torch.load(model_path))
        model.eval()

        xmin, xmax = UtilityFunctions.find_min_max (train_h_folder)

        val_dataset = Ds.CustData(val_h_folder, xmin, xmax)    
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


        no_of_save = 1
        no_of_generations_per_sample = 60
        idx = 0

        for hbsl, hrot, hboth in val_loader:

            h_src_path = os.path.join (out_folder, str(idx) + "_h_src.mat")
            h_dest_path = os.path.join (out_folder, str(idx) + "_h_dest.mat")

            scipy.io.savemat (h_src_path, \
                mdict={'h_src': np.squeeze(hbsl.detach().cpu().numpy())})

            scipy.io.savemat (h_dest_path, \
                mdict={'h_dest': np.squeeze(hrot.detach().cpu().numpy())})

            #save h_init, h_f
            hbsl = hbsl.to(device)
            hrot = hrot.to(device)
            hboth = hboth.to(device)

            
            hbsl = torch.unsqueeze (hbsl, 0)
            hbsl = torch.repeat_interleave (hbsl, no_of_generations_per_sample, 0)

            xs, z_det = model.det_encoder(hbsl) 

            # z_det = torch.repeat_interleave (z_det, no_of_generations_per_sample, 0)
            
            # i = 0
            # while i < len (xs):
            #     xs[i] = torch.repeat_interleave (xs[i], no_of_generations_per_sample, 0)
            #     i += 1
            
            z_var = torch.randn((no_of_generations_per_sample, latent_dim)).to(device)
            z_var = torch.clip(z_var, -2.5, 2.5)
            z = torch.cat((z_det, z_var),  dim=1)

            recon_hrot = model.decoder(z, xs, no_of_generations_per_sample)

            h_recon_path = os.path.join (out_folder, str(idx) + "_h_recon_z_change.mat")

            scipy.io.savemat (h_recon_path, \
                mdict={'h_recon': np.squeeze(recon_hrot.detach().cpu().numpy())})

            idx += 1

            hrot = torch.repeat_interleave (hrot, no_of_generations_per_sample, 0)
            hrot = hrot.unsqueeze(1)

            print ("HBSL size = ", hbsl.size())
            print ("H_RECO SIZE = ", recon_hrot.size())
            print ("Rand H size = ", hrot.size() )

            print ("Diff between Src and Generations = ",\
                 torch.linalg.norm((recon_hrot - hbsl).flatten()))


            print ("Diff between Random H and Generations = ",\
                 torch.linalg.norm((recon_hrot - hrot).flatten()))

            if idx == no_of_save:
                break

            

    @staticmethod
    def reconstruction_blockvarenc(model_path,encoder_path, out_folder):

        """
        Generate by blocking out varenc 
            - First generate varenc mapping where h_i and h_f are same 
            - For different h_i's use the mapping from step 1 
        """


        model = PHER()
        model = model.to(device)

        model.load_state_dict(torch.load(model_path))
        model.eval()

        var_encoder = Var_Encoder()
        var_encoder = var_encoder.to(device)


        var_encoder.load_state_dict(torch.load(encoder_path))
        var_encoder.eval()


        xmin, xmax = UtilityFunctions.find_min_max (train_h_folder)

        val_dataset = Ds.CustData('Data/Compare_Points/Val_folder', xmin, xmax)    
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

        no_of_generations_per_sample = 2

        for hbsl, hrot, hboth in val_loader:

            hbsl = hbsl.to(device)

            
            hbsl = torch.unsqueeze (hbsl, 1)

            org_hbsl = hbsl
            org_hbsl = torch.repeat_interleave (org_hbsl, no_of_generations_per_sample, 0)

            hbsl = torch.cat((hbsl, hbsl) , axis = 1)
            mu_no_diff, logvar_no_diff =  var_encoder(hbsl)
            z_var_no_diff = var_encoder.reparameterize (mu_no_diff, logvar_no_diff)
            np.savetxt(os.path.join(out_folder, 'z_var_no_diff.txt'),\
                 z_var_no_diff.detach().cpu().numpy())
            break

        val_loader = DataLoader(val_dataset, batch_size=no_of_generations_per_sample, shuffle=True)
        #print ("Z var no diff", z_var_no_diff)
        z_var_no_diff = torch.repeat_interleave (z_var_no_diff, no_of_generations_per_sample, 0)

        for hbsl, hrot, hboth in val_loader:

            hbsl = hbsl.to(device)
            hrot = hrot.to(device)
            hboth = hboth.to(device)

            #hbsl = torch.unsqueeze (hbsl, 1)
            #xs, z_det = model.det_encoder(hbsl)
            mu, var = var_encoder (hboth)
            z_var = var_encoder.reparameterize (mu, var)
            #z = torch.cat((z_det, z_var_no_diff),  dim=1)

            z, recon_hrot = model(hbsl, z_var_no_diff)
            #recon_hrot = model.decoder(z, xs, no_of_generations_per_sample)


            h_src_path = os.path.join (out_folder, "h_src.mat")
            h_recon_path = os.path.join (out_folder, "h_recon_no_diff.mat")
            h_dest_path = os.path.join (out_folder, "h_dest.mat")
            
            scipy.io.savemat (h_src_path, \
                mdict={'h_src': np.squeeze(hbsl.detach().cpu().numpy())})


            scipy.io.savemat (h_recon_path, \
                mdict={'h_recon': np.squeeze(recon_hrot.detach().cpu().numpy())})

            scipy.io.savemat (h_dest_path, \
                mdict={'h_dest': np.squeeze(hrot.detach().cpu().numpy())})


            org_hbsl = org_hbsl.squeeze(1)

            print ("H SRC SIZE = ", hbsl.size())
            print ("H Dest SIZE = ", hrot.size())
            print ("H RECON SIZE = ", recon_hrot.size())
            print ("H ORG SIZE ", org_hbsl.size())


            print ("h_src diff", torch.linalg.norm((recon_hrot - hbsl).flatten()) )
            print ("h_dest diff", torch.linalg.norm((recon_hrot - hrot).flatten()))
            print ("org bsl diff", torch.linalg.norm((recon_hrot - org_hbsl).flatten()))
            break


    @staticmethod
    def plot_z_var (encoder_path, plot_out):

        var_encoder = Var_Encoder()
        var_encoder = var_encoder.to(device)


        var_encoder.load_state_dict(torch.load(encoder_path))
        var_encoder.eval()


        xmin, xmax = UtilityFunctions.find_min_max (train_h_folder)

        val_dataset = Ds.CustData(val_h_folder, xmin, xmax)    
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        z_ep = [] 

        for hbsl, hrot, hboth in val_loader:

            hboth = hboth.to(device)
            hbsl = hbsl.to(device)
            hbsl = hbsl.unsqueeze (1)
            hbsl_comb = torch.cat ((hbsl, hbsl), 1)
            mu, logvar = var_encoder (hbsl_comb)
            z_var = var_encoder.reparameterize (mu, logvar)

            for latent_vector in z_var.detach().cpu().numpy():
                z_ep.append (latent_vector)

        fig, axs = plt.subplots(4, 4, sharey=True, tight_layout=True)
        i_fig = 0
        while (i_fig < 4):
            j_fig = 0
            while j_fig < 4:
                axs[i_fig][j_fig].hist (np.array(z_ep)[:,j_fig + 4*i_fig], bins=10) 
                j_fig += 1
            i_fig += 1
        plt.savefig (os.path.join(plot_out, 'Z_dist.png'))


    @staticmethod
    def loss_som(input_vector,som):

        z_bmu=torch.zeros_like(input_vector)
        som_weights=som.get_weights()
        for j in range(len(input_vector)):
            w = som.winner(input_vector[j].detach().cpu().numpy())
            z_bmu[j,:]=torch.from_numpy(som_weights[w])
            #print ("Winner = ", w)
        
        MSE = F.mse_loss(input_vector, z_bmu,reduction="sum")
        #MSE = torch.mean(torch.sum(F.mse_loss(input_vector, z_bmu,reduction="Nome"),dim=[1])) \
            #originally none but changed to sum to match with rest of the losses
 
        return MSE

    @staticmethod
    def save_som(filename,som):
        with open(filename, 'wb') as outfile:
            pickle.dump(som, outfile)


    @staticmethod
    def encoded_mappings (encoder_path, nb_of_samples = 1000, use_train_set=False, full_dataset=False):
        
        local_batch_size = 128
        
        num = 0

        assert (nb_of_samples%local_batch_size == 0)


        var_encoder = Var_Encoder()
        var_encoder = var_encoder.to(device)

        var_encoder.load_state_dict(torch.load(encoder_path))
        var_encoder.eval()


        xmin, xmax = UtilityFunctions.find_min_max (train_h_folder)

        if use_train_set:
            val_dataset = Ds.CustData(train_h_folder, xmin, xmax, True)    
        
        else:
            val_dataset = Ds.CustData(val_h_folder, xmin, xmax, True)


        if full_dataset:
            nb_of_samples = len (val_dataset)

        encodings = np.zeros((nb_of_samples,latent_dim))
        error_labels = np.zeros((nb_of_samples))

        val_loader = DataLoader(val_dataset, batch_size=local_batch_size, shuffle=True,\
            drop_last= True)


        for idx, (hbsl, hrot, hboth, label) in enumerate(val_loader):

            hboth = hboth.to(device)
            mu, logvar = var_encoder (hboth)
            z_var = var_encoder.reparameterize (mu, logvar)
            
            encodings[num:num + local_batch_size] = z_var.detach().cpu().numpy().squeeze()
            error_labels[num:num + local_batch_size] = label.numpy().squeeze()

            num += local_batch_size

            
            if num == nb_of_samples:
                return encodings, error_labels

            

            
            
        return encodings, error_labels

    
    @staticmethod
    def extract_operation_values (name):

        a = re.search('_xt(.*)_yt', name)
        a = a.group(1)
        b = re.search('_yt(.*)_zt', name)
        b = b.group(1)
        c = re.search('_zt(.*).mat', name)
        c = c.group(1)
        d = re.search('_zrot(.*)_xt', name)
        d = d.group(1)
        e = re.search('_yrot(.*)_zrot', name)
        e = e.group(1)
        f = re.search('_xrot(.*)_yrot', name)
        f = f.group(1)

        labels = [a, b, c, f, e, d] #xt,yt,zt,xr,yr,zr

        return labels



    @staticmethod    
    def extract_labels (h_bsl_name, h_dest_name):

        bsl_labels = UtilityFunctions.extract_operation_values (h_bsl_name)
        dest_labels = UtilityFunctions.extract_operation_values (h_dest_name)

        no_of_operators = 6
        [0 for _ in range (no_of_operators)]

        label_str = ""

        for i in range (no_of_operators):
            if bsl_labels[i] == dest_labels[i]:
                label_str += '0'
            else:
                label_str += '1'

        return label_str

    def VAE_Baseline_Generations_Recons (model_path, gen_out_path, recon_out_path):

        N = 60
        model = Baseline_VAE().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        z_var = torch.randn((N,latent_dim)).to(device)
        generations = model.decoder(z_var,N)

        generations = generations.detach().cpu().numpy().squeeze()

        scipy.io.savemat (gen_out_path, \
                mdict={'h_gen': generations})

        xmin, xmax = UtilityFunctions.find_min_max (train_h_folder)

        val_dataset = BsD.Baseline_Dataset(val_h_folder, xmin, xmax)    
        val_loader = DataLoader(val_dataset, batch_size=N, shuffle=True)

        for H in val_loader:

            H = H.to(device)
            z, recon, mu, logvar = model (H)

            print ('Recon loss  = ', torch.norm((recon.squeeze() - H.squeeze()).flatten() ))

            #print (recon.size())
            recon = recon.unsqueeze(1)

            
            
            final = torch.cat ((H, recon), dim = 1)
            
            
            final = final.detach().cpu().numpy().squeeze()

            scipy.io.savemat (recon_out_path, \
                mdict={'h_recon': final})

            break

            

if __name__ == "__main__":

    # UtilityFunctions.plot_z_var \
    #     ('Model_Weights/SOM_0.01_New_errors_var_encoder_16_epoch1950_latent0.001_lr0.0001_reg0.01', \
    #         'img' )
    
    
    UtilityFunctions.reconstruction_blockvarenc\
        ('Model_Weights/SOM_Cluster_only_New_errors_modl_no_elu_last_detenc_500_16_epoch1950_latent0.001_lr0.0001_reg0.01',\
        'Model_Weights/SOM_Cluster_only_New_errors_var_encoder_16_epoch1950_latent0.001_lr0.0001_reg0.01',\
            'img/block_var')
    
    # UtilityFunctions.reconstruction_zchange(\
    #     'Model_Weights/SOM_0.01_New_errors_modl_no_elu_last_detenc_500_16_epoch1950_latent0.001_lr0.0001_reg0.01',\
    #         'img/z_change')


    # UtilityFunctions.VAE_Baseline_Generations_Recons (\
    #     'Model_Weights/Baseline_VAE_New_errors_modl_no_elu_last_detenc_500_16_epoch1950_latent0.1_lr0.0001',\
    #         gen_out_path = 'img/baseline/gen.mat', recon_out_path= 'img/baseline/recon.mat' )






