import torch
from torch.utils.data import Dataset

import os
import scipy
from Constants import H, W, operator_labels
import re
import numpy as np
import Utils
#from Utils import UtilityFunctions

class CustData(Dataset):


    def __init__(self,h_folder, xmin, xmax, return_label=False, write_names=False):

        self.xmin = xmin
        self.xmax = xmax
        self.return_label = return_label
        self.write_names = write_names

        self.h_folder = h_folder
        self.h_list = os.listdir (h_folder)
        self.total_h = len (self.h_list)

        self.h_pairs = []

        for h1 in self.h_list:
            for h2 in self.h_list:
                #if h1 != h2: #just for optimization
                self.h_pairs.append ((h1,h2))
        
        print ("Total H Pairs = ", len (self.h_pairs))

        # self.H_bsl_path = H_bsl_path
        # self.H_dest_path = H_dest_path
        # self.train = train_test
        # self.bsl = os.listdir(H_bsl_path)
        # self.dest = os.listdir(H_dest_path)
    def __len__(self):
        return len(self.h_pairs)


    def __getitem__(self, idx):

        H_bsl_name, H_dest_name = self.h_pairs[idx]

        H_bsl_path = os.path.join(self.h_folder, H_bsl_name)
        matFile = scipy.io.loadmat(H_bsl_path)
        h_bsl = matFile['H']
        normalized_b = (h_bsl-self.xmin)/(self.xmax-self.xmin)
        h_bsl = torch.from_numpy(normalized_b).float()

        H_dest_path = os.path.join(self.h_folder, H_dest_name)
        matFile = scipy.io.loadmat(H_dest_path)
        h_dest = matFile['H']
        normalized_dest = (h_dest-self.xmin)/(self.xmax-self.xmin)
        h_dest = torch.from_numpy(normalized_dest).float()
        

        h_both = torch.cat((h_bsl, h_dest)).view(2, H, W)
    
        #return h_bsl, h_dest, h_both, heart_bsl, heart_dest, bsl_name, dest_name
        
        H_names = [H_bsl_name, H_dest_name]
        #print ("Names = ", H_names)

        if self.write_names:
        
            H_names = [H_bsl_name, H_dest_name]
            names_filename = 'ECGI_logs_sim_data/H_names.txt'
            answ=os.path.exists(names_filename)

            with open(names_filename, "a" if answ else "w") as myfile:
                myfile.write(str(H_names) + '\n')

        if self.return_label:
            if 'scale' in H_bsl_name or 'scale' in H_dest_name:
                return h_bsl, h_dest, h_both, -1
            label = Utils.UtilityFunctions.extract_labels (H_bsl_name, H_dest_name)
            label = operator_labels[label]
            return h_bsl, h_dest, h_both, label

        return h_bsl, h_dest, h_both


