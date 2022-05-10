from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


from Constants import latent_dim, H, W

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, lambda_val, beta_val=1.0):
        ctx.lambda_val = lambda_val
        ctx.beta_val = beta_val
        print ("BETA ------------------------", beta_val)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg_() * ctx.lambda_val * ctx.beta_val
        return output, None, None




class PHER(nn.Module):
    def __init__(self):
        super(PHER, self).__init__()
        
        #UNET
        ##########encoder###########
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 8), stride=(3, 4), padding=(1, 4))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 5), stride=(2, 2), padding=(2, 1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 4), stride=(2, 3) , padding=(1, 1))        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2) , padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1) , padding=(0, 0))

        self.fc7 = nn.Linear(256*6*10, 1024)
        self.fc8 = nn.Linear(1024, 128)
        self.fc9 = nn.Linear(128, 32)
        self.fc10 = nn.Linear(32, latent_dim)


        ##########unet decoder###########
        self.fcd1 = nn.Linear(2*latent_dim, 64)
        self.fcd2 = nn.Linear(64, 1024)
        self.fcd3 = nn.Linear(1024, 128*6*10)

        self.dconv8 = nn.ConvTranspose2d(384, 256, kernel_size=(1, 1), stride=(1, 1) , padding=(0, 0))
        self.dconv7 = nn.ConvTranspose2d(384, 128, kernel_size=(3, 3), stride=(2, 2) , padding=(1, 1))
        self.dconv6 = nn.ConvTranspose2d(192, 64, kernel_size=(3, 4), stride=(2, 3) , padding=(1, 1)) 
        self.dconv5 = nn.ConvTranspose2d(96, 32,  kernel_size=(4, 5), stride=(2, 2), padding=(2, 1))
        self.dconv4 = nn.ConvTranspose2d(48, 8, kernel_size=(5, 8), stride=(3, 4), padding=(1, 4))
        self.dconv3 = nn.ConvTranspose2d(8, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))




    def det_encoder(self, image):
        x1 = F.elu(self.conv1(image))
        x2 = F.elu(self.conv2(x1))
        
        x3 = F.elu(self.conv3(x2))
        
        x4 = F.elu(self.conv4(x3))
        
        x5 = F.elu(self.conv5(x4))
        
        x6 = F.elu(self.fc7(torch.flatten(x5, 1)))
        x7 = F.elu(self.fc8(torch.flatten(x6, 1)))
        x8 = F.elu(self.fc9(torch.flatten(x7, 1)))
        x9 = F.elu(self.fc10(torch.flatten(x8, 1)))
    

        xs = [x1, x2, x3, x4, x5]

        return xs, x9


    def decoder(self, z_unet, xs, N):
        x = self.fcd1(z_unet)
        x = F.elu (x)
        x = self.fcd2(x)
        x = F.elu (x)
        x = self.fcd3(x)
        x = F.elu (x)
        x = x.view(N, -1, 6, 10)
        
        x = F.elu(self.dconv8(torch.cat((x, xs[4]) , 1)))
        x = F.elu(self.dconv7(torch.cat((x, xs[3]) , 1)))
        x = F.elu(self.dconv6(torch.cat((x, xs[2]) , 1)))
        x = F.elu(self.dconv5(torch.cat((x, xs[1]) , 1)))
        x = F.elu(self.dconv4(torch.cat((x, xs[0]) , 1)))
        reconx = self.dconv3(x)

        return reconx



    def forward(self, bsl, z_var):
        N, h, w = bsl.shape
        bsl = bsl.view(N, -1, h, w)
        #rot = rot.view(N, -1, h, w)
        xs, z_det = self.det_encoder(bsl)
        z = torch.cat((z_det, z_var),  dim=1)
        reconx = self.decoder(z, xs, N)
        reconx = reconx.view(N, H, W)
        return z, reconx

    def call_dec(self, z_var, z_det, xs):
        N = 1
        z = torch.cat((z_det, z_var.view(1,-1)),  dim=1)
        reconx = self.decoder(z, xs, N)
        reconx = reconx.view(N, H, W)
        return reconx


    def generate_op(self, bsl):
        h, w = bsl.shape
        N = 1
        bsl = bsl.view(N, -1, h, w)
        xs, z_det = self.det_encoder(bsl)
        z_var = torch.rand_like(z_det)
        z = torch.cat((z_det, z_var),  dim=1)
        reconx = self.decoder(z, xs, N)
        reconx = reconx.view(N, H, W)
        return z_det, z_var, z, xs, reconx

    def generate_blockskip(self, hbsl, hrot, hboth, xi, i):
        N, h, w = hbsl.shape
        hbsl = hbsl.view(N, -1, h, w)
        hrot = hrot.view(N, -1, h, w)
        xs, z_det = self.det_encoder(hbsl)
        mu, logvar = self.var_encoder(hboth)
        z_var = self.reparameterize(mu, logvar)
        z = torch.cat((z_det, z_var),  dim=1)
        for k in xi:
            xs[k] = torch.zeros_like(xs[k])
        reconx = self.decoder(z, xs, N)
        reconx = reconx[i].view(H, W)
        return reconx



class Var_Encoder (nn.Module):

    def __init__(self):
        super(Var_Encoder, self).__init__()

        ##########p encoder###########
        self.convp1 = nn.Conv2d(2, 8, kernel_size=(5, 8), stride=(3, 4), padding=(1, 4))
        self.convp2 = nn.Conv2d(8, 16, kernel_size=(4, 5), stride=(2, 2), padding=(2, 1))
        self.convp3 = nn.Conv2d(16, 32, kernel_size=(3, 4), stride=(2, 3) , padding=(1, 1)) 
        self.convp4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2) , padding=(1, 1))
        self.convp5 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1) , padding=(0, 0))

        self.fcp7 = nn.Linear(128*6*10, 1024)
        self.fcp8 = nn.Linear(1024, 128)
        self.fcp9 = nn.Linear(128, 32)
        self.fcp101 = nn.Linear(32, latent_dim)
        self.fcp102 = nn.Linear(32, latent_dim)


    def forward (self, hboth):

        x1 = F.elu(self.convp1(hboth))
        x2 = F.elu(self.convp2(x1))
        x3 = F.elu(self.convp3(x2))
        x4 = F.elu(self.convp4(x3))
        x5 = F.elu(self.convp5(x4))
        x6 = F.elu(self.fcp7(torch.flatten(x5, 1)))
        x7 = F.elu(self.fcp8(x6))
        x8 = F.elu(self.fcp9(x7))        
        x9_pmu = self.fcp101(x8)
        x9_psigma = self.fcp102(x8)

        return x9_pmu, x9_psigma


    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = eps.mul (std)
        z = z + mu
        return z



class Adv_Decoder(nn.Module):
    def __init__(self):
        super(Adv_Decoder, self).__init__()


        ########adversarial decoder#############
        self.adv_fcd1 = nn.Linear(latent_dim, 64)
        self.adv_fcd2 = nn.Linear(64, 1024)
        self.adv_fcd3 = nn.Linear(1024, 128*6*10)

        self.adv_dconv8 = nn.ConvTranspose2d(128, 64, kernel_size=(1, 1), stride=(1, 1) , padding=(0, 0))
        self.adv_dconv7 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2) , padding=(1, 1))
        self.adv_dconv6 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 4), stride=(2, 3) , padding=(1, 1)) 
        self.adv_dconv5 = nn.ConvTranspose2d(16, 8,  kernel_size=(4, 5), stride=(2, 2), padding=(2, 1))
        self.adv_dconv4 = nn.ConvTranspose2d(8, 4, kernel_size=(5, 8), stride=(3, 4), padding=(1, 4))
        self.adv_dconv3 = nn.ConvTranspose2d(4, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))


    def forward (self,z_var, lambda_val=0.0, beta_val=200.0):

        """
        Recieve z_var and generate h_f 
        the loss is output adversarial_dec(z_var) - h_f
        """
        N = z_var.size()[0]

        x = ReverseLayerF.apply(z_var, lambda_val, beta_val)

        x = self.adv_fcd1(x)
        x = F.elu (x)
        x = self.adv_fcd2(x)
        x = F.elu (x)
        x = self.adv_fcd3(x)
        x = F.elu (x)
        x = x.view(N, -1, 6, 10)
        
        x = F.elu(self.adv_dconv8(x))
        x = F.elu(self.adv_dconv7(x))
        x = F.elu(self.adv_dconv6(x))
        x = F.elu(self.adv_dconv5(x))
        x = F.elu(self.adv_dconv4(x))
        reconx = self.adv_dconv3(x)
        
        
        return reconx


    



