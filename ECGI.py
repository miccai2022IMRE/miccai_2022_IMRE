import imp
import numpy as np
import torch
from numpy import linalg as LA
import pybobyqa
import Constants as ps
from torch.utils.data import DataLoader
from Models.PHER import PHER, Var_Encoder
from Models.Baseline_VAE import Baseline_VAE
from Utils import UtilityFunctions
import os
import scipy
from torch.utils.tensorboard import SummaryWriter
from Constants import latent_dim, epochs, lr, beta, train_h_folder, val_h_folder, \
    BATCH_SIZE, device, model_dir, log_dir, beta, H as Height, W as Width
from Dataset import CustData
from Utils import UtilityFunctions
import Utils as ut
import pickle
from SOM_Classification import classify

def generate_bsp(H, xmin, xmax, SNR, egm, option):
    eg_len = egm.shape[1]
    I = np.ones((ps.H, ps.W))
    H = H.detach().cpu().numpy()*(xmax -xmin) + I*xmin
    mea = H.dot(egm)
    [dimz,t] = mea.shape
    sig_avg = np.mean(mea ** 2)
    sig_avg_db = 10 * np.log10(sig_avg)
    noise_avg_db = sig_avg_db - SNR
    noise_avg = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg), size=(dimz, t))
    bsp = mea + noise_volts
   
    return bsp

def tikhonov(h, L , y, lam):
    L_lam = lam * L
    all_len = ps.W
    precision = h.transpose().dot(h) + L_lam
    u, s, v = LA.svd(precision, full_matrices=True)
    var = s
    v = v.transpose()
    Cov=v[:,0:all_len - 1].dot(np.diag(1/var[0:all_len-1])).dot((u[:,0:all_len-1]).transpose())
    x = Cov.dot(h.transpose().dot(y))

    return x

def minim_func(z_vae, model, x, skips, bsp, z_geo, h_max, h_min):
   
    f = LA.norm(np.squeeze(model.decoder(torch.cat((z_geo, torch.from_numpy(z_vae).to(device)\
        .reshape((1, ps.latent_dim)).float().to(device)) , dim=1), skips, 1).detach().cpu().numpy()\
            *(h_max - h_min) + h_min).dot(x) - bsp)
    return f


def minim_func_baseline(z_var, model, x, bsp, h_max, h_min):

    z_var = torch.from_numpy(z_var).to(device).float()
    H_out = model.decoder (z_var, 1)
    H_out = H_out.detach().cpu().numpy().squeeze()

    H_out = H_out*(h_max - h_min) + h_min
    out = H_out.dot(x)
    f = LA.norm (out - bsp) 
   
    # f = LA.norm(np.squeeze(model.decoder(torch.cat((z_geo, torch.from_numpy(z_vae).to(device)\
    #     .reshape((1, ps.latent_dim)).float().to(device)) , dim=1), skips, 1).detach().cpu().numpy()\
    #         *(h_max - h_min) + h_min).dot(x) - bsp)
    return f


def ECGI(model, H_grt, H_in, egm, bsp, L, Height, Width, hmax, hmin):
  
    model.eval()
        
    skips, z_Hin = model.det_encoder(H_in.reshape(1, 1, Height, Width).to(ps.device)) 
    
    z_var = torch.randn_like(z_Hin)
    z = torch.cat((z_Hin, z_var),  dim=1)

    Hz = model.decoder(z, skips, 1)*(hmax -hmin) + hmin
    Hz = Hz *(hmax -hmin) + hmin


    
    z_var = np.squeeze(z_var.detach().cpu().numpy())
    z_var = z_var.flatten()
    a = z_var.shape

 

    loopnum = 15

    H_opt = Hz
    lam = 0.005
    lower = -5 * np.ones((a))
    upper = 5* np.ones((a))
    
    x = tikhonov(H_in.detach().cpu().numpy().reshape(Height, Width)*(hmax - hmin) + hmin, L , bsp, lam)
    # z_var = torch.from_numpy(z_var).to(ps.device).reshape((1, -1))
    # z_var = z_var.float().to(ps.device) 
    import logging
    for j in range(loopnum):
        print("enter main loop at iter" + str(j))
        if j < 30:
            lam = 0.005
            # print(x.shape)
        logging.basicConfig(level=logging.INFO, format='%(message)s')

        soln = pybobyqa.solve(minim_func, z_var, args = (model, x, skips, bsp, z_Hin, hmax, hmin), print_progress=True,\
                              bounds = (lower , upper), seek_global_minimum = True,\
                              objfun_has_noise=False, rhobeg = 5, maxfun=100)
        z_var = soln.x
        #print ("Found Z Var", z_var)
        #np.savetxt ('ECGI_logs_rho_0.1/z_var.txt',z_var)

        
        z_var = torch.from_numpy(z_var).to(ps.device).reshape((1, -1))
        z_var = z_var.float().to(ps.device) 
        z = torch.cat((z_Hin, z_var),  dim=1)
        H_opt = model.decoder(z, skips, 1) *(hmax -hmin) + hmin
        H_opt = np.squeeze(H_opt.detach().cpu().numpy())
        H_opt = H_opt.reshape(Height, Width)

        

        x = tikhonov(H_opt, L , bsp, lam)

        z_var = z_var.detach().cpu().numpy()
        # z_var = z_var.astype('float64')
        z_var = z_var.flatten()

    #np.savetxt ('ECGI_logs_sim_data/'+ str() +'_z_var.txt',z_var)
    H_in = np.squeeze(H_in.detach().cpu().numpy())
    H_in = H_in.reshape(Height, Width)*(hmax - hmin) + hmin
    x_init = tikhonov(H_in, L , bsp, lam)
    
    
    H_grt = np.squeeze(H_grt.detach().cpu().numpy())
    H_grt = H_grt.reshape(Height, Width)*(hmax - hmin) + hmin
    x_grt = tikhonov(H_grt, L , bsp, lam)
    
    return x_init, x, H_in, H_opt, H_grt, x_grt, z_var



def ECGI_baseline (model, H_grt, H_in, egm, bsp, L, Height, Width, hmax, hmin):
  
    model.eval()

    H_in  = H_in.unsqueeze(0).unsqueeze(0)
    print ('Hin size = ', H_in.size())
    mu, logvar = model.encoder (H_in)
    z_var = model.reparameterize (mu, logvar)

    
    z_var = np.squeeze(z_var.detach().cpu().numpy())
    z_var = z_var.flatten()
    a = z_var.shape

 

    loopnum = 15

    lam = 0.005
    lower = -5 * np.ones((a))
    upper = 5* np.ones((a))
    
    x = tikhonov(H_in.detach().cpu().numpy().reshape(Height, Width)*(hmax - hmin) + hmin, L , bsp, lam)
    # z_var = torch.from_numpy(z_var).to(ps.device).reshape((1, -1))
    # z_var = z_var.float().to(ps.device) 
    import logging
    for j in range(loopnum):
        print("enter main loop at iter" + str(j))
        if j < 30:
            lam = 0.005
            # print(x.shape)
        logging.basicConfig(level=logging.INFO, format='%(message)s')


        soln = pybobyqa.solve(minim_func_baseline, z_var, args = (model, x, bsp, hmax, hmin), print_progress=True,\
                              bounds = (lower , upper), seek_global_minimum = True,\
                              objfun_has_noise=False, rhobeg = 5, maxfun=100)
        z_var = soln.x
        print ("Found Z Var", z_var)
        #np.savetxt ('ECGI_logs_rho_0.1/z_var.txt',z_var)

        
        z_var = torch.from_numpy(z_var).to(ps.device).reshape((1, -1))
        z_var = z_var.float().to(ps.device) 
        H_opt = model.decoder(z_var, 1) *(hmax -hmin) + hmin
        H_opt = np.squeeze(H_opt.detach().cpu().numpy())
        H_opt = H_opt.reshape(Height, Width)

        

        x = tikhonov(H_opt, L , bsp, lam)

        z_var = z_var.detach().cpu().numpy()
        z_var = z_var.flatten()


    H_in = np.squeeze(H_in.detach().cpu().numpy())
    H_in = H_in.reshape(Height, Width)*(hmax - hmin) + hmin
    x_init = tikhonov(H_in, L , bsp, lam)
    
    
    H_grt = np.squeeze(H_grt.detach().cpu().numpy())
    H_grt = H_grt.reshape(Height, Width)*(hmax - hmin) + hmin
    x_grt = tikhonov(H_grt, L , bsp, lam)
    
    return x_init, x, H_in, H_opt, H_grt, x_grt






def ECGI_real_data (model, H_in, egm, bsp, L, Height, Width, hmax, hmin):
  

    encoder_path = 'Model_Weights/SOM_Cluster_only_New_errors_var_encoder_16_epoch1950_latent0.001_lr0.0001_reg0.01'
    var_encoder = Var_Encoder ()
    var_encoder.load_state_dict(torch.load(encoder_path))
    var_encoder = var_encoder.to(device)
    var_encoder.eval()


    sample_H = 'Data/Random_Split/Train_H/H_xrot0_yrot0_zrot0_xt0_yt0_zt11.mat'
    sample_H = scipy.io.loadmat (sample_H)
    sample_H = sample_H['H']

    sample_H = torch.from_numpy (sample_H)
    sample_H = sample_H.float()
    sample_H = sample_H.to(device)
    sample_H = torch.unsqueeze (sample_H, 0)

    sample_H = (sample_H-hmin)/(hmax-hmin)


    sample_H = torch.cat((sample_H, sample_H), dim=0)
    sample_H = torch.unsqueeze (sample_H, 0)

    mu, logvar = var_encoder (sample_H)
    z_var = var_encoder.reparameterize (mu, logvar)

    model.eval()

    H_in = torch.from_numpy(H_in)
    H_in = H_in.float()
    H_in = H_in.to(device)
        
    skips, z_Hin = model.det_encoder(H_in.reshape(1, 1, Height, Width)) 
    
    
    z = torch.cat((z_Hin, z_var),  dim=1)

    Hz = model.decoder(z, skips, 1)

    Hz = torch.squeeze (Hz)
    # print ("H SRC SIZE = ", H_in.size())
    # print ("H RECON SIZE = ", Hz.size())
    # print ("H ORG SIZE ", sample_H[0][0].size())


    # print ("h_src diff", torch.linalg.norm((Hz - H_in).flatten()) )
    # print ("org bsl diff", torch.linalg.norm((Hz - sample_H[0][0]).flatten()))


    # scipy.io.savemat ('H_var_extract.mat', \
    #     mdict={'h_var': np.squeeze(sample_H[0][0].detach().cpu().numpy())})

    # scipy.io.savemat ('H_real_init.mat', \
    #             mdict={'h_src': np.squeeze(H_in.detach().cpu().numpy())})


    # scipy.io.savemat ('H_real_recon.mat', \
    #     mdict={'h_recon': np.squeeze(Hz.detach().cpu().numpy())})



    Hz = Hz *(hmax -hmin) + hmin


    
    z_var = np.squeeze(z_var.detach().cpu().numpy())
    z_var = z_var.flatten()
    a = z_var.shape

 

    loopnum = 15 #5 

    H_opt = Hz
    lam = 0.005
    lower = -5 * np.ones((a))
    upper = 5* np.ones((a))
    
    x = tikhonov(H_in.detach().cpu().numpy().reshape(Height, Width)*(hmax - hmin) + hmin, L , bsp, lam)
    # z_var = torch.from_numpy(z_var).to(ps.device).reshape((1, -1))
    # z_var = z_var.float().to(ps.device) 
    import logging
    for j in range(loopnum):
        print("enter main loop at iter" + str(j))
        if j < 30:
            lam = 0.005 #0.8 * lam
            # print(x.shape)
        logging.basicConfig(level=logging.INFO, format='%(message)s')

        soln = pybobyqa.solve(minim_func, z_var, args = (model, x, skips, bsp, z_Hin, hmax, hmin), print_progress=True,\
                              bounds = (lower , upper), seek_global_minimum = True,\
                              objfun_has_noise=False, rhobeg= 5, maxfun=100)
        z_var = soln.x
        print ("Found Z Var", z_var)
        #np.savetxt ('ECGI_logs_rho_0.1/z_var.txt',z_var)

        
        z_var = torch.from_numpy(z_var).to(ps.device).reshape((1, -1))
        z_var = z_var.float().to(ps.device) 
        z = torch.cat((z_Hin, z_var),  dim=1)
        H_opt = model.decoder(z, skips, 1) *(hmax -hmin) + hmin
        H_opt = np.squeeze(H_opt.detach().cpu().numpy())
        H_opt = H_opt.reshape(Height, Width)

        

        x = tikhonov(H_opt, L , bsp, lam)

        z_var = z_var.detach().cpu().numpy()
        # z_var = z_var.astype('float64')
        z_var = z_var.flatten()


    H_in = np.squeeze(H_in.detach().cpu().numpy())
    H_in = H_in.reshape(Height, Width)*(hmax - hmin) + hmin
    x_init = tikhonov(H_in, L , bsp, lam)
    
    
    # H_grt = np.squeeze(H_grt.detach().cpu().numpy())
    # H_grt = H_grt.reshape(Height, Width)*(hmax - hmin) + hmin
    # x_grt = tikhonov(H_grt, L , bsp, lam)
    
    return x_init, x, H_in, H_opt

 


def ECGI_optimize(path, model_name):

    # with open('Model_Weights/New_errors_som_cluster_only.p', 'rb') as infile:
    #     som = pickle.load(infile)  

    optimized_z_vars = []



    local_batch_size = 20
    model = PHER()
    #out = 'models_dir/' + model_name
    out = os.path.join (model_dir, model_name)
    model.load_state_dict(torch.load(out))
    model.eval().to(device)
    # eval_vae(test_loader, m)

    xmin, xmax = UtilityFunctions.find_min_max(train_h_folder)
    
    egm_path = 'Data/Compare_Points/EGMs'
    #'/home/stu3/s15/nk4856/Research/ForwardAdaptation/Data/EGMs/'
    egms = os.listdir(egm_path)
    #egms = egms[:4] #for now
    SNR = 35
    
    matFile = scipy.io.loadmat('Data/sim_L.mat')
    L = matFile['L']
    
    all_mse = [0, 0]
    all_scc = [0, 0]
    all_tcc = [0, 0]
    
    log_file = path + 'log_ecgi.txt'
    count = 0
    batch_count = 0

    val_dataset = CustData('Data/Compare_Points/Val_folder', xmin, xmax, write_names = False)
    val_loader = DataLoader(val_dataset, batch_size=local_batch_size, shuffle=False)

    with open(log_file, 'w+') as f:
        for j, egm in enumerate(egms):
            
            
            egm = os.path.join(egm_path, egm)
            egm = scipy.io.loadmat(egm)
            egm = egm['U_surf_sig']
            for hbsl, hrot, hboth in val_loader:
                
            
                hbsl = hbsl.to(device)
                hrot = hrot.to(device)
                
                for i in range(local_batch_size):
                    count += 1
                    bsp = generate_bsp(hrot[i], xmin, xmax, SNR, egm, 1)
                    x_init, x, H_in, H_opt, H_grt, x_grt, z_var = ECGI(model, hrot[i], hbsl[i], egm, bsp, L, Height, Width, xmax, xmin)
                    #if count % 1 == 0:

                    optimized_z_vars.append (list(z_var))
                    scipy.io.savemat(path + 'x_init_' + str(count) + '.mat', mdict={'x_in': x_init})
                    scipy.io.savemat(path + 'x_' + str(count) + '.mat', mdict={'x': x})
                    scipy.io.savemat(path + 'egm_' + str(count) + '.mat', mdict={'egm': egm})
                    scipy.io.savemat(path + 'bsp_' + str(count) + '.mat', mdict={'bsp': bsp})
                    scipy.io.savemat(path + 'H_grt_' + str(count) + '.mat', mdict={'H_grt': H_grt})
                    scipy.io.savemat(path + 'H_init_' + str(count) + '.mat', mdict={'H_in': H_in})
                    scipy.io.savemat(path + 'H_opt_' + str(count) + '.mat', mdict={'H_opt': H_opt})
                    scipy.io.savemat(path + 'x_grt' + str(count) + '.mat', mdict={'x_grt': x_grt})
                
                    rmse = ut.UtilityFunctions.rmse_er(egm, x)
                    scc = ut.UtilityFunctions.space_cc(egm, x)
                    tcc = ut.UtilityFunctions.temp_cc(egm, x)
                    all_mse[0] += rmse
                    all_scc[0] += scc
                    all_tcc[0] += tcc
                    
                    rmse_in = ut.UtilityFunctions.rmse_er(egm, x_init)
                    scc_in = ut.UtilityFunctions.space_cc(egm, x_init)
                    tcc_in = ut.UtilityFunctions.temp_cc(egm, x_init)
                    all_mse[1] += rmse_in
                    all_scc[1] += scc_in
                    all_tcc[1] += tcc_in
                    
                    log = 'base rmse {:.6f}  scc {}  tcc {}  opt_rmse {:.6f}  scc {}  tcc {} '.format(rmse_in, scc_in, tcc_in, rmse, scc, tcc)
                    f.writelines(log) 
                    f.write("\n")


        np.savetxt ('ECGI_logs_sim_data/'+ str() +'_z_var.txt',np.array(optimized_z_vars))
        
        f.writelines('total samples {:04d}: '.format(count))
        f.writelines('total rmse error on initial x is {:.4f}: '.format(all_mse[1]/count))
        f.writelines('total scc error on initial x is {:.4f}: '.format(all_scc[1]/count))
        f.writelines('total tcc error on initial x is {:.4f}: '.format(all_tcc[1]/count))
        f.writelines('total rmse error on x is {:.4f}: '.format(all_mse[0]/count))
        f.writelines('total scc error on x is {:.4f}: '.format(all_scc[0]/count))
        f.writelines('total tcc error on x is {:.4f}: '.format(all_tcc[0]/count))




def ECGI_optimize_real_data (path, model_name):
    model = PHER()
    out = os.path.join (model_dir, model_name)
    
    model.load_state_dict(torch.load(out))
    model.eval().to(device)
    # eval_vae(test_loader, m)

    xmin, xmax = UtilityFunctions.find_min_max(train_h_folder)
    
    egm_path = 'Data/real_data_processed/real_data_processed/egms'
    #'/home/stu3/s15/nk4856/Research/ForwardAdaptation/Data/EGMs/'
    egms = os.listdir(egm_path)
    #egms = egms[:1] #for now
    
    SNR = 35


    bsp_path = 'Data/real_data_processed/real_data_processed/bsps'
    
    matFile = scipy.io.loadmat('Data/real_data_processed/real_data_processed/h_L.mat')
    L = matFile['h_L']

    
    all_mse = [0, 0]
    all_scc = [0, 0]
    all_tcc = [0, 0]
    
    log_file = path + 'log_ecgi.txt'
    count = 1
    batch_count = 0

    real_hbsl = scipy.io.loadmat ('Data/real_data_processed/real_data_processed/H.mat')
    hbsl = real_hbsl['H']


    hbsl = (hbsl-xmin)/(xmax-xmin)




    #val_dataset = CustData(val_h_folder, xmin, xmax)
    #val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with open(log_file, 'w+') as f:
        for i, egm_name in enumerate(egms):
            
                egm = os.path.join(egm_path, egm_name)
                egm = scipy.io.loadmat(egm)
                egm = egm['egm']

                bsp_name = egm_name.replace ('egm', 'bsp')
                bsp_complete_path = os.path.join (bsp_path, bsp_name)
                bsp = scipy.io.loadmat (bsp_complete_path)
                bsp = bsp['bsp']


                x_init, x, H_in, H_opt = ECGI_real_data(model, hbsl, egm, bsp, L, Height, Width, xmax, xmin)
                
                scipy.io.savemat(path + 'x_init_' + str(i) + '.mat', mdict={'x_in': x_init})
                scipy.io.savemat(path + 'x_' + str(i) + '.mat', mdict={'x': x})
                scipy.io.savemat(path + 'egm_' + str(i) + '.mat', mdict={'egm': egm})
                scipy.io.savemat(path + 'bsp_' + str(i) + '.mat', mdict={'bsp': bsp})
                scipy.io.savemat(path + 'H_init_' + str(i) + '.mat', mdict={'H_in': H_in})
                scipy.io.savemat(path + 'H_opt_' + str(i) + '.mat', mdict={'H_opt': H_opt})
                    
                
                rmse = ut.UtilityFunctions.rmse_er(egm, x)
                scc = ut.UtilityFunctions.space_cc(egm, x)
                tcc = ut.UtilityFunctions.temp_cc(egm, x)
                all_mse[0] += rmse
                all_scc[0] += scc
                all_tcc[0] += tcc
                
                rmse_in = ut.UtilityFunctions.rmse_er(egm, x_init)
                scc_in = ut.UtilityFunctions.space_cc(egm, x_init)
                tcc_in = ut.UtilityFunctions.temp_cc(egm, x_init)
                all_mse[1] += rmse_in
                all_scc[1] += scc_in
                all_tcc[1] += tcc_in
                
                log = 'base rmse {:.6f}  scc {}  tcc {}  opt_rmse {:.6f}  scc {}  tcc {} '.format(rmse_in, scc_in, tcc_in, rmse, scc, tcc)
                f.writelines(log) 
                f.write("\n")
        
        f.writelines('total samples {:04d}: '.format(count))
        f.writelines('total rmse error on initial x is {:.4f}: '.format(all_mse[1]/count))
        f.writelines('total scc error on initial x is {:.4f}: '.format(all_scc[1]/count))
        f.writelines('total tcc error on initial x is {:.4f}: '.format(all_tcc[1]/count))
        f.writelines('total rmse error on x is {:.4f}: '.format(all_mse[0]/count))
        f.writelines('total scc error on x is {:.4f}: '.format(all_scc[0]/count))
        f.writelines('total tcc error on x is {:.4f}: '.format(all_tcc[0]/count))



def ECGI_optimize_human_real_data (path, model_name):

    model = PHER()
    out = os.path.join (model_dir, model_name)
    
    model.load_state_dict(torch.load(out))
    model.eval().to(device)
    # eval_vae(test_loader, m)

    xmin, xmax = UtilityFunctions.find_min_max(train_h_folder)
    
    egm_path = 'Data/real_data_processed/real_data_processed/egms'
    #'/home/stu3/s15/nk4856/Research/ForwardAdaptation/Data/EGMs/'
    egms = os.listdir(egm_path)
    egms = egms[:1] #for now
    
    SNR = 35


    bsp_complete_path = 'Data/New_real_data_processed/New_real_data_processed/33_for_opt/bsp_lv.mat'
    
    matFile = scipy.io.loadmat('Data/New_real_data_processed/New_real_data_processed/33_for_opt/L.mat')
    L = matFile['L']

    
    all_mse = [0, 0]
    all_scc = [0, 0]
    all_tcc = [0, 0]
    
    log_file = path + 'log_ecgi.txt'
    count = 1
    batch_count = 0

    real_hbsl = scipy.io.loadmat ('Data/New_real_data_processed/New_real_data_processed/33_for_opt/H.mat')
    hbsl = real_hbsl['H']


    hbsl = (hbsl-xmin)/(xmax-xmin)




    #val_dataset = CustData(val_h_folder, xmin, xmax)
    #val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with open(log_file, 'w+') as f:
        for i, egm_name in enumerate(egms):
            
                egm = os.path.join(egm_path, egm_name)
                egm = scipy.io.loadmat(egm)
                egm = egm['egm']

                bsp = scipy.io.loadmat (bsp_complete_path)
                bsp = bsp['bsp']


                x_init, x, H_in, H_opt = ECGI_real_data(model, hbsl, egm, bsp, L, Height, Width, xmax, xmin)
                
                scipy.io.savemat(path + 'x_init_' + str(i) + '.mat', mdict={'x_in': x_init})
                scipy.io.savemat(path + 'x_' + str(i) + '.mat', mdict={'x': x})
                #scipy.io.savemat(path + 'egm_' + str(i) + '.mat', mdict={'egm': egm})
                scipy.io.savemat(path + 'bsp_' + str(i) + '.mat', mdict={'bsp': bsp})
                scipy.io.savemat(path + 'H_init_' + str(i) + '.mat', mdict={'H_in': H_in})
                scipy.io.savemat(path + 'H_opt_' + str(i) + '.mat', mdict={'H_opt': H_opt})
                    
                
                rmse = ut.UtilityFunctions.rmse_er(egm, x)
                scc = ut.UtilityFunctions.space_cc(egm, x)
                tcc = ut.UtilityFunctions.temp_cc(egm, x)
                all_mse[0] += rmse
                all_scc[0] += scc
                all_tcc[0] += tcc
                
                rmse_in = ut.UtilityFunctions.rmse_er(egm, x_init)
                scc_in = ut.UtilityFunctions.space_cc(egm, x_init)
                tcc_in = ut.UtilityFunctions.temp_cc(egm, x_init)
                all_mse[1] += rmse_in
                all_scc[1] += scc_in
                all_tcc[1] += tcc_in
                
                log = 'base rmse {:.6f}  scc {}  tcc {}  opt_rmse {:.6f}  scc {}  tcc {} '.format(rmse_in, scc_in, tcc_in, rmse, scc, tcc)
                f.writelines(log) 
                f.write("\n")
        
        f.writelines('total samples {:04d}: '.format(count))
        f.writelines('total rmse error on initial x is {:.4f}: '.format(all_mse[1]/count))
        f.writelines('total scc error on initial x is {:.4f}: '.format(all_scc[1]/count))
        f.writelines('total tcc error on initial x is {:.4f}: '.format(all_tcc[1]/count))
        f.writelines('total rmse error on x is {:.4f}: '.format(all_mse[0]/count))
        f.writelines('total scc error on x is {:.4f}: '.format(all_scc[0]/count))
        f.writelines('total tcc error on x is {:.4f}: '.format(all_tcc[0]/count))
                            
                            
 

def ECGI_optimize_baseline (path, model_name):


    local_batch_size = 72
    model = Baseline_VAE()
    #out = 'models_dir/' + model_name
    out = os.path.join (model_dir, model_name)
    model.load_state_dict(torch.load(out))
    model.eval().to(device)
    # eval_vae(test_loader, m)

    xmin, xmax = UtilityFunctions.find_min_max(train_h_folder)
    
    egm_path = 'Data/Compare_Points/EGMs'
    #'/home/stu3/s15/nk4856/Research/ForwardAdaptation/Data/EGMs/'
    egms = os.listdir(egm_path)
    #egms = egms[:4] #for now
    SNR = 35
    
    matFile = scipy.io.loadmat('/home/stu3/s15/nk4856/Research/ForwardAdaptation/Data/sim_L.mat')
    L = matFile['L']
    
    all_mse = [0, 0]
    all_scc = [0, 0]
    all_tcc = [0, 0]
    
    log_file = path + 'log_ecgi.txt'
    count = 0
    batch_count = 0

    val_dataset = CustData('Data/Compare_Points/Border_H', xmin, xmax, write_names = True)
    val_loader = DataLoader(val_dataset, batch_size=local_batch_size, shuffle=False)

    with open(log_file, 'w+') as f:
        for j, egm in enumerate(egms):
            
            
            egm = os.path.join(egm_path, egm)
            egm = scipy.io.loadmat(egm)
            egm = egm['U_surf_sig']
            for hbsl, hrot, hboth in val_loader:
                
            
                hbsl = hbsl.to(device)
                hrot = hrot.to(device)
                
                for i in range(local_batch_size):
                    count += 1
                    bsp = generate_bsp(hrot[i], xmin, xmax, SNR, egm, 1)
                    x_init, x, H_in, H_opt, H_grt, x_grt = ECGI_baseline(model, hrot[i], hbsl[i], egm, bsp, L, Height, Width, xmax, xmin)
                    #if count % 1 == 0:
                    scipy.io.savemat(path + 'x_init_' + str(count) + '.mat', mdict={'x_in': x_init})
                    scipy.io.savemat(path + 'x_' + str(count) + '.mat', mdict={'x': x})
                    scipy.io.savemat(path + 'egm_' + str(count) + '.mat', mdict={'egm': egm})
                    scipy.io.savemat(path + 'bsp_' + str(count) + '.mat', mdict={'bsp': bsp})
                    scipy.io.savemat(path + 'H_grt_' + str(count) + '.mat', mdict={'H_grt': H_grt})
                    scipy.io.savemat(path + 'H_init_' + str(count) + '.mat', mdict={'H_in': H_in})
                    scipy.io.savemat(path + 'H_opt_' + str(count) + '.mat', mdict={'H_opt': H_opt})
                    scipy.io.savemat(path + 'x_grt' + str(count) + '.mat', mdict={'x_grt': x_grt})
                
                    rmse = ut.UtilityFunctions.rmse_er(egm, x)
                    scc = ut.UtilityFunctions.space_cc(egm, x)
                    tcc = ut.UtilityFunctions.temp_cc(egm, x)
                    all_mse[0] += rmse
                    all_scc[0] += scc
                    all_tcc[0] += tcc
                    
                    rmse_in = ut.UtilityFunctions.rmse_er(egm, x_init)
                    scc_in = ut.UtilityFunctions.space_cc(egm, x_init)
                    tcc_in = ut.UtilityFunctions.temp_cc(egm, x_init)
                    all_mse[1] += rmse_in
                    all_scc[1] += scc_in
                    all_tcc[1] += tcc_in
                    
                    log = 'base rmse {:.6f}  scc {}  tcc {}  opt_rmse {:.6f}  scc {}  tcc {} '.format(rmse_in, scc_in, tcc_in, rmse, scc, tcc)
                    f.writelines(log) 
                    f.write("\n")
        
        f.writelines('total samples {:04d}: '.format(count))
        f.writelines('total rmse error on initial x is {:.4f}: '.format(all_mse[1]/count))
        f.writelines('total scc error on initial x is {:.4f}: '.format(all_scc[1]/count))
        f.writelines('total tcc error on initial x is {:.4f}: '.format(all_tcc[1]/count))
        f.writelines('total rmse error on x is {:.4f}: '.format(all_mse[0]/count))
        f.writelines('total scc error on x is {:.4f}: '.format(all_scc[0]/count))
        f.writelines('total tcc error on x is {:.4f}: '.format(all_tcc[0]/count))



def ECGI_optimize_baseline_real_data (path, model_name):


    model = Baseline_VAE()
    #out = 'models_dir/' + model_name
    out = os.path.join (model_dir, model_name)
    model.load_state_dict(torch.load(out))
    model.eval().to(device)
    # eval_vae(test_loader, m)

    xmin, xmax = UtilityFunctions.find_min_max(train_h_folder)
    
    egm_path = 'Data/real_data_processed/real_data_processed/egms'
    #'/home/stu3/s15/nk4856/Research/ForwardAdaptation/Data/EGMs/'
    egms = os.listdir(egm_path)
    #egms = egms[:1] #for now
    
    SNR = 35


    bsp_path = 'Data/real_data_processed/real_data_processed/bsps'
    
    matFile = scipy.io.loadmat('Data/real_data_processed/real_data_processed/h_L.mat')
    L = matFile['h_L']

    
    all_mse = [0, 0]
    all_scc = [0, 0]
    all_tcc = [0, 0]
    
    log_file = path + 'log_ecgi.txt'
    count = 1
    batch_count = 0

    real_hbsl = scipy.io.loadmat ('Data/real_data_processed/real_data_processed/H_xp60.mat')
    hbsl = real_hbsl['H']


    hbsl = (hbsl-xmin)/(xmax-xmin)

    hbsl = torch.from_numpy (hbsl).to(device).float()


    #val_dataset = CustData(val_h_folder, xmin, xmax)
    #val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with open(log_file, 'w+') as f:
        for i, egm_name in enumerate(egms):
            
                egm = os.path.join(egm_path, egm_name)
                egm = scipy.io.loadmat(egm)
                egm = egm['egm']

                bsp_name = egm_name.replace ('egm', 'bsp')
                bsp_complete_path = os.path.join (bsp_path, bsp_name)
                bsp = scipy.io.loadmat (bsp_complete_path)
                bsp = bsp['bsp']


                x_init, x, H_in, H_opt = ECGI_baseline (model, None ,hbsl, egm, bsp, L, Height, Width, xmax, xmin)
                
                scipy.io.savemat(path + 'x_init_' + str(i) + '.mat', mdict={'x_in': x_init})
                scipy.io.savemat(path + 'x_' + str(i) + '.mat', mdict={'x': x})
                scipy.io.savemat(path + 'egm_' + str(i) + '.mat', mdict={'egm': egm})
                scipy.io.savemat(path + 'bsp_' + str(i) + '.mat', mdict={'bsp': bsp})
                scipy.io.savemat(path + 'H_init_' + str(i) + '.mat', mdict={'H_in': H_in})
                scipy.io.savemat(path + 'H_opt_' + str(i) + '.mat', mdict={'H_opt': H_opt})
                    
                
                rmse = ut.UtilityFunctions.rmse_er(egm, x)
                scc = ut.UtilityFunctions.space_cc(egm, x)
                tcc = ut.UtilityFunctions.temp_cc(egm, x)
                all_mse[0] += rmse
                all_scc[0] += scc
                all_tcc[0] += tcc
                
                rmse_in = ut.UtilityFunctions.rmse_er(egm, x_init)
                scc_in = ut.UtilityFunctions.space_cc(egm, x_init)
                tcc_in = ut.UtilityFunctions.temp_cc(egm, x_init)
                all_mse[1] += rmse_in
                all_scc[1] += scc_in
                all_tcc[1] += tcc_in
                
                log = 'base rmse {:.6f}  scc {}  tcc {}  opt_rmse {:.6f}  scc {}  tcc {} '.format(rmse_in, scc_in, tcc_in, rmse, scc, tcc)
                f.writelines(log) 
                f.write("\n")
        
        f.writelines('total samples {:04d}: '.format(count))
        f.writelines('total rmse error on initial x is {:.4f}: '.format(all_mse[1]/count))
        f.writelines('total scc error on initial x is {:.4f}: '.format(all_scc[1]/count))
        f.writelines('total tcc error on initial x is {:.4f}: '.format(all_tcc[1]/count))
        f.writelines('total rmse error on x is {:.4f}: '.format(all_mse[0]/count))
        f.writelines('total scc error on x is {:.4f}: '.format(all_scc[0]/count))
        f.writelines('total tcc error on x is {:.4f}: '.format(all_tcc[0]/count))



def ECGI_optimize_baseline_real_human_data (path, model_name):


    model = Baseline_VAE()
    #out = 'models_dir/' + model_name
    out = os.path.join (model_dir, model_name)
    model.load_state_dict(torch.load(out))
    model.eval().to(device)
    # eval_vae(test_loader, m)

    xmin, xmax = UtilityFunctions.find_min_max(train_h_folder)
    
    egm_path = 'Data/real_data_processed/real_data_processed/egms'
    #'/home/stu3/s15/nk4856/Research/ForwardAdaptation/Data/EGMs/'
    egms = os.listdir(egm_path)
    egms = egms[:1] #for now
    
    SNR = 35


    bsp_complete_path = 'Data/New_real_data_processed/New_real_data_processed/24_for_opt/bsp_rv.mat'
    
    matFile = scipy.io.loadmat('Data/New_real_data_processed/New_real_data_processed/24_for_opt/L.mat')
    L = matFile['L']

    
    all_mse = [0, 0]
    all_scc = [0, 0]
    all_tcc = [0, 0]
    
    log_file = path + 'log_ecgi.txt'
    count = 1
    batch_count = 0

    real_hbsl = scipy.io.loadmat ('Data/New_real_data_processed/New_real_data_processed/24_for_opt/H.mat')
    hbsl = real_hbsl['H']


    hbsl = (hbsl-xmin)/(xmax-xmin)

    hbsl = torch.from_numpy (hbsl).to(device).float()

    #val_dataset = CustData(val_h_folder, xmin, xmax)
    #val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with open(log_file, 'w+') as f:
        for i, egm_name in enumerate(egms):
            
                egm = os.path.join(egm_path, egm_name)
                egm = scipy.io.loadmat(egm)
                egm = egm['egm']

                bsp = scipy.io.loadmat (bsp_complete_path)
                bsp = bsp['bsp']


                x_init, x, H_in, H_opt = ECGI_baseline(model, None ,hbsl, egm, bsp, L, Height, Width, xmax, xmin)
                
                scipy.io.savemat(path + 'x_init_' + str(i) + '.mat', mdict={'x_in': x_init})
                scipy.io.savemat(path + 'x_' + str(i) + '.mat', mdict={'x': x})
                #scipy.io.savemat(path + 'egm_' + str(i) + '.mat', mdict={'egm': egm})
                scipy.io.savemat(path + 'bsp_' + str(i) + '.mat', mdict={'bsp': bsp})
                scipy.io.savemat(path + 'H_init_' + str(i) + '.mat', mdict={'H_in': H_in})
                scipy.io.savemat(path + 'H_opt_' + str(i) + '.mat', mdict={'H_opt': H_opt})
                    
                
                rmse = ut.UtilityFunctions.rmse_er(egm, x)
                scc = ut.UtilityFunctions.space_cc(egm, x)
                tcc = ut.UtilityFunctions.temp_cc(egm, x)
                all_mse[0] += rmse
                all_scc[0] += scc
                all_tcc[0] += tcc
                
                rmse_in = ut.UtilityFunctions.rmse_er(egm, x_init)
                scc_in = ut.UtilityFunctions.space_cc(egm, x_init)
                tcc_in = ut.UtilityFunctions.temp_cc(egm, x_init)
                all_mse[1] += rmse_in
                all_scc[1] += scc_in
                all_tcc[1] += tcc_in
                
                log = 'base rmse {:.6f}  scc {}  tcc {}  opt_rmse {:.6f}  scc {}  tcc {} '.format(rmse_in, scc_in, tcc_in, rmse, scc, tcc)
                f.writelines(log) 
                f.write("\n")
        
        f.writelines('total samples {:04d}: '.format(count))
        f.writelines('total rmse error on initial x is {:.4f}: '.format(all_mse[1]/count))
        f.writelines('total scc error on initial x is {:.4f}: '.format(all_scc[1]/count))
        f.writelines('total tcc error on initial x is {:.4f}: '.format(all_tcc[1]/count))
        f.writelines('total rmse error on x is {:.4f}: '.format(all_mse[0]/count))
        f.writelines('total scc error on x is {:.4f}: '.format(all_scc[0]/count))
        f.writelines('total tcc error on x is {:.4f}: '.format(all_tcc[0]/count))



   






if __name__ == "__main__":

    ECGI_optimize ('ECGI_logs_sim_data/',\
        'SOM_Cluster_only_New_errors_modl_no_elu_last_detenc_500_16_epoch1950_latent0.001_lr0.0001_reg0.01')