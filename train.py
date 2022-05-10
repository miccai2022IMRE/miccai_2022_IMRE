import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Models.PHER import PHER, Var_Encoder
from Utils import UtilityFunctions
import os
import torch.nn.functional as F
import numpy as np
from minisom import MiniSom    
import pickle

import datetime
from torch.utils.tensorboard import SummaryWriter

from Constants import latent_dim, epochs, lr, beta, train_h_folder, val_h_folder, \
    BATCH_SIZE, device, model_dir, log_dir, reg_lambda, include_som, som_constant
from Dataset import CustData

def train_vae():


    if include_som:
        som_hw = 40
        som = MiniSom(som_hw, som_hw, latent_dim, sigma=1.5, learning_rate=.7, activation_distance='euclidean',
              topology='hexagonal', neighborhood_function='gaussian', random_seed=10)
        
    else:
        som = None
    
    log_folder =  os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter (log_folder)
    
    recon_train_a, kl_train_a, recon_train_Geo, theta_train_a, total_train_a = [], [], [], [], []
    recon_test_a, kl_test_a, recon_test_Geo, theta_test_a, total_test_a = [], [], [], [], []

    model = PHER()
    #model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    var_encoder = Var_Encoder ()
    #var_encoder.load_state_dict(torch.load(encoder_path))
    var_encoder = var_encoder.to(device)

    best_val_loss = 1000000

    xmin, xmax = UtilityFunctions.find_min_max (train_h_folder)

    train_dataset = CustData(train_h_folder, xmin, xmax)
    val_dataset = CustData(val_h_folder, xmin, xmax)

    print ("Train Examples = ", len(train_dataset)) 

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    
    file_name = "model_final1_" + str(epochs) +  "lat" + str(latent_dim) + "lr" + str(lr) + ".txt"
    with open( file_name, 'a') as f:
        for epoch in range(epochs):
            # if epoch> 1800:
            #     beta = 0.005
            hrecon, kltrain, total, epoch_reg_loss, epoch_som_loss\
                 = train(epoch, beta, train_loader, model, var_encoder, som)
            hrecon_test, kltest, total_test, reg_loss_test = test(beta, val_loader, model, var_encoder)
            hrecon /= len(train_dataset)
            epoch_reg_loss /= len(train_dataset)
            epoch_som_loss /= len (train_dataset)
            kltrain /= len(train_dataset)
            total /= len(train_dataset)
            hrecon_test /= len(val_dataset)

            kltest /= len(val_dataset)
            total_test /= len(val_dataset)
            reg_loss_test /= len(val_dataset)


            log = '[E {:03d}] Trn: {:.6f}, TrhRec: {:.6f}, kltr: {:.4f}, Tst: {:.4f},\
                 TsHtrc: {:.6f}, kltest: {:.4f}, SOM: {:.4f}'\
                     .format(epoch , total, hrecon, kltrain, total_test,hrecon_test, kltest, epoch_som_loss)
            f.writelines(log) 
            f.write("\n")
            f.flush()
            recon_train_a.append(hrecon)
            kl_train_a.append(kltrain)
            total_train_a.append(total)

            recon_test_a.append(hrecon_test)
            kl_test_a.append(kltest)
            total_test_a.append(total_test)

            writer.add_scalar ("loss/train_recon", hrecon, epoch)
            writer.add_scalar ("loss/train_kld", kltrain, epoch)
            writer.add_scalar ("loss/train_total", total, epoch)
            writer.add_scalar ("loss/train_reg", epoch_reg_loss, epoch)

            writer.add_scalar ("loss/val_reg",reg_loss_test, epoch )
            writer.add_scalar ("loss/val_recon", hrecon_test, epoch)
            writer.add_scalar ("loss/val_kld", kltest, epoch)
            writer.add_scalar ("loss/val_total", total_test, epoch)


            if total_test <= best_val_loss:
                torch.save(model.state_dict(), model_dir + '/SOM_Cluster_New_errors_modl_no_elu_last_detenc_500_{}_epoch{}_latent{}_lr{}_reg{}'.format(latent_dim, epochs, beta, lr, reg_lambda))
                torch.save (var_encoder.state_dict(),model_dir + '/SOM_Cluster_New_errors_var_encoder_{}_epoch{}_latent{}_lr{}_reg{}'.format(latent_dim, epochs, beta, lr, reg_lambda) )

                if som:
                    UtilityFunctions.save_som(model_dir + '/New_errors_som_0.01_only.p', som)

                best_val_loss = total_test


    # plot_losses(recon_train_a, recon_test_a, 'reconstruction', args.epoch)
    # plot_losses(kl_train_a, kl_test_a, 'latent', args.epoch)
    # plot_losses(total_train_a, total_test_a, 'total', args.epoch)
    # plot_losses(recon_train_Geo, recon_test_Geo, 'geo', args.epoch)



def train(epoch , beta, train_loader, model, var_encoder, som):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    var_optimizer = optim.Adam (var_encoder.parameters(), lr=lr)


    model.train()
    var_encoder.train()
    hrecon, grecon, latent, total_loss, theta_er, kl_er, epoch_reg_loss, epoch_som_loss\
         = 0, 0, 0, 0, 0, 0, 0, 0
    n = 0

    for hbsl, hrot, hboth in train_loader:
        hbsl = hbsl.to(device)
        hrot = hrot.to(device)
        hboth = hboth.to(device)


        optimizer.zero_grad()
        var_optimizer.zero_grad()

        mu, logvar = var_encoder (hboth)
        z_var = var_encoder.reparameterize (mu, logvar)
        z, recon_hrot = model(hbsl, z_var)


        mse_H , klpos, total = UtilityFunctions.loss_function(hrot, recon_hrot, mu, logvar, beta)

        if som:
            loss_som=UtilityFunctions.loss_som(mu,som)
            loss_som = loss_som * som_constant
            total = total + loss_som
            epoch_som_loss  += loss_som.item()



        total.backward()
        hrecon += mse_H.item()
        kl_er += klpos.item()
        
        total_loss += total.item()

        optimizer.step()
        var_optimizer.step()

        if som:
            #print ("Training SOM ")
            som.train_random(mu.cpu().detach().numpy(), 10, verbose=False)


        optimizer.zero_grad()
        var_optimizer.zero_grad()


        hrot = torch.unsqueeze (hrot, 1)
        same_hs = torch.cat((hrot, hrot), axis = 1)

        mu_same, logvar_same = var_encoder (same_hs)
        z_var_same = var_encoder.reparameterize(mu_same, logvar_same)
        z, recon_reg = model (hbsl, z_var_same)

        

        reg_loss = F.mse_loss(hbsl, recon_reg, reduction='sum')
        reg_loss = reg_loss * reg_lambda #0.1 (still not going to dest) - 0.
    
        reg_loss.backward()
        epoch_reg_loss += reg_loss.item()

        
        optimizer.step()
        var_optimizer.step()

        total_loss += reg_loss.item()


        n += 1
        log = '[E {:03d}] Loss: {:.4f}, HRec loss: {:.4f}, kl-pos: {:.4f}, Reg Loss: {:.4f}, SOM loss: {:.4f}'\
            .format(epoch, total_loss / (n * BATCH_SIZE), hrecon / (n * BATCH_SIZE),\
                 kl_er / (n * BATCH_SIZE), epoch_reg_loss / (n * BATCH_SIZE), epoch_som_loss / (n * BATCH_SIZE))
        #UtilityFunctions.inline_print(log)
        print (log)
    
    
    return hrecon, kl_er, total_loss, epoch_reg_loss, epoch_som_loss


def test(beta, val_loader, model, var_encoder):
    model.eval()
    var_encoder.eval()
    hrecon, grecon, total_loss, theta_er, kl_er, epoch_reg_loss = 0, 0, 0, 0, 0, 0
    n = 0
    for hbsl, hrot, hboth in val_loader:
        hbsl = hbsl.to(device)
        hrot = hrot.to(device)
        hboth = hboth.to(device)

        mu, logvar = var_encoder (hboth)
        z_var = var_encoder.reparameterize (mu, logvar)
        z, recon_hrot = model(hbsl, z_var)


        mse_H , klpos, total = UtilityFunctions.loss_function(hrot, recon_hrot, mu, logvar, beta)

        hrot = torch.unsqueeze (hrot, 1)
        same_hs = torch.cat((hrot, hrot), axis = 1)

        mu_same, logvar_same = var_encoder (same_hs)
        z_var_same = var_encoder.reparameterize(mu_same, logvar_same)
        z, recon_reg = model (hbsl, z_var_same)

        reg_loss = F.mse_loss(hbsl, recon_reg.squeeze(1), reduction='sum')
        reg_loss = reg_loss * reg_lambda
        

        epoch_reg_loss += reg_loss.item()
        hrecon += mse_H.item()
        
        kl_er += klpos.item()
        total_loss += total.item()
        total_loss += reg_loss.item()
        
        n += 1

    return hrecon, kl_er, total_loss, epoch_reg_loss



