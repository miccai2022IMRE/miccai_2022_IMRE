
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 64
W = 448
H = 120
lr = 1e-4
latent_dim = 16

include_som = True
som_constant = 0
reg_lambda = 0.02
theta_beta = 1
geo_beta = 1
pos_b = 0.01
opt_num = 0
maxnumb = 500
beta = 0.001
baseline_beta = 0.1
epochs = 1950

# Prepare dataset
train_h_folder = 'Data/Random_Split/Train_H'


val_h_folder = 'Data/Random_Split/Val_H'
#'Data/Compare_Points/Val_folder'

model_dir = "Model_Weights"
log_dir = "logs"


operator_labels = {
    '000000':0,
    '100000':1,
    '010000':2,
    '001000':3,
    '000100':4,
    '000010':5,
    '000001':6,
    '110000':7,
    '101000':8,
    '100100':9,
    '100010':10,
    '100001':11,
    '011000':12,
    '010100':13,
    '010010':14,
    '010001':15,
    '001100':16,
    '001010':17,
    '001001':18,
    '000110':19,
    '000101':20,
    '000011':21
}

error_label_names = ['ne', 'xt', 'yt', 'zt', 'xr', 'yr', 'zr', 'xtyt', 'xtzt', 'xtxr', 'xtyr',\
    'xtzr', 'ytzt', 'ytxr', 'ytyr', 'ytzr', 'ztxr', 'ztyr', 'ztzr', 'xryr', 'xrzr', 'yrzr']