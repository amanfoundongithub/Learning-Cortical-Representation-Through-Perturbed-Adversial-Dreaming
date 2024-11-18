
from __future__ import print_function
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from utils import *
from network import *


epsilon = 0.0 # amount of noise in wake latent space

##########################################################

niter = 50    # Iterations 
R = 1       # REM 
W = 1        # Wake
N = 1        # NREM

lmbd = 0.5    # convex combination factor for REM

dir_files = "./result/cifar100/model_wnf" 
try:
    os.makedirs(dir_files)
except OSError:
    pass

dir_checkpoint = "./checkpoint/cifar100/model_wnf"
try:
    os.makedirs(dir_checkpoint)
except OSError:
    pass

##########################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset, unorm, img_channels = get_dataset("cifar10", "./cifar10", 32)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle=True, drop_last=True)

# some hyper parameters
nz = 256
batch_size = 64

# setup networks
netG = Generator(ngpu = 1, nz = nz, ngf = 64, img_channels = img_channels)
netG.apply(weights_init)
netD = Discriminator(ngpu = 1, nz = nz, ndf = 64, img_channels = img_channels,  p_drop = 0.0)
netD.apply(weights_init)
# send to GPU
netD.to(device)
netG.to(device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr = 2e-4, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 2e-4, betas=(0.5, 0.999))
d_losses = []
g_losses = []
r_losses_real = []
r_losses_fake = []
kl_losses = []


# loss functions
dis_criterion = nn.BCELoss() # discriminator

# Reconstructor loss function 
rec_criterion = nn.MSELoss() # reconstruction

# tensor placeholders
dis_label = torch.zeros(batch_size, dtype=torch.float32, device=device)

# Define labels for real and fake data 
real_label_value = 1.0
fake_label_value = 0

# Evaluation noise for performing evaluation 
eval_noise = torch.randn(batch_size, nz, device=device)

for epoch in range(niter):

    store_loss_D = []
    store_loss_G = []
    store_loss_R_real = []
    store_loss_R_fake = []
    store_norm = []
    store_kl = []

    for i, data in enumerate(dataloader, 0):

        ############################
        # Wake (W)
        ###########################
        
        # Discrimination wake
        optimizerD.zero_grad()
        optimizerG.zero_grad()
        real_image, label = data
        real_image, label = real_image.to(device), label.to(device)
        
        # Latent output, discriminator output 
        latent_output, dis_output = netD(real_image)
        
        # Add noise if any
        latent_output_noise = latent_output + epsilon * torch.randn(batch_size, nz, device=device) 
        
        # Since we are awake, we should assign a real label 1 to it 
        dis_label[:] = real_label_value 
        
        # Now backpropogate the loss 
        dis_errD_real = dis_criterion(dis_output, dis_label)
        if R > 0.0: 
            (dis_errD_real).backward(retain_graph=True)

        # KL divergence regularization
        kl = kl_loss(latent_output)
        (kl).backward(retain_graph=True)
        
        # reconstruction Real data space
        reconstructed_image = netG(latent_output_noise, reverse=False)
        rec_real = rec_criterion(reconstructed_image, real_image)
        if W > 0.0:
            (W*rec_real).backward()
        optimizerD.step()
        optimizerG.step()
        # compute the mean of the discriminator output (between 0 and 1)
        D_x = dis_output.cpu().mean()
        latent_norm = torch.mean(torch.norm(latent_output.squeeze(), dim=1)).item()
        
        
        ###########################
        # NREM perturbed dreaming (N)
        ##########################
        optimizerD.zero_grad()
        latent_z = latent_output.detach()
        
        with torch.no_grad():
            nrem_image = netG(latent_z)
            occlusion = Occlude(drop_rate=random.random(), tile_size=random.randint(1,8))
            occluded_nrem_image = occlusion(nrem_image, d=1)
        latent_recons_dream, _ = netD(occluded_nrem_image)
        rec_fake = rec_criterion(latent_recons_dream, latent_output.detach())
        if N > 0.0:
            (N * rec_fake).backward()
        optimizerD.step()

     
        ###########################
        # REM adversarial dreaming (R)
        ##########################

        optimizerD.zero_grad()
        optimizerG.zero_grad()
        lmbd = lmbd 
        noise = torch.randn(batch_size, nz, device=device)
        if i==0:
            latent_z = 0.5*latent_output.detach() + 0.5*noise
        else:
            latent_z = 0.25*latent_output.detach() + 0.25*old_latent_output + 0.5*noise
        
        dreamed_image_adv = netG(latent_z, reverse=True)
        latent_recons_dream, dis_output = netD(dreamed_image_adv)
        dis_label[:] = fake_label_value 
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        
        if R > 0.0: 
            dis_errD_fake.backward(retain_graph=True)
            optimizerD.step()
            optimizerG.step()
        dis_errG = - dis_errD_fake

        D_G_z1 = dis_output.cpu().mean()

        old_latent_output = latent_output.detach()
        
        
        ###########################
        # Compute average losses
        ###########################
        store_loss_G.append(dis_errG.item())
        store_loss_D.append((dis_errD_fake + dis_errD_real).item())
        store_loss_R_real.append(rec_real.item())
        store_loss_R_fake.append(rec_fake.item())
        store_norm.append(latent_norm)
        store_kl.append(kl.item())
        


        if i % 200 == 0 and i>1:
            print('[%d/%d][%d/%d]  Loss_D: %.4f  Loss_G: %.4f  Loss_R_real: %.4f  Loss_R_fake: %.4f  D(x): %.4f  D(G(z)): %.4f  latent_norm : %.4f  '
                % (epoch, niter, i, len(dataloader),
                    np.mean(store_loss_D), np.mean(store_loss_G), np.mean(store_loss_R_real), np.mean(store_loss_R_fake), D_x, D_G_z1, np.mean(latent_norm) ))
            compare_img_rec = torch.zeros(batch_size * 2, real_image.size(1), real_image.size(2), real_image.size(3))
            with torch.no_grad():
                reconstructed_image = netG(latent_output)
            compare_img_rec[::2] = real_image
            compare_img_rec[1::2] = reconstructed_image
            vutils.save_image(unorm(compare_img_rec[:128]), '%s/recon_%03d.png' % (dir_files, epoch), nrow=8)
            fake = unorm(dreamed_image_adv)
            vutils.save_image(fake[:64].data, '%s/fake_%03d.png' % (dir_files, epoch), nrow=8)
            

    d_losses.append(np.mean(store_loss_D))
    g_losses.append(np.mean(store_loss_G))
    r_losses_real.append(np.mean(store_loss_R_real))
    r_losses_fake.append(np.mean(store_loss_R_fake))
    kl_losses.append(np.mean(store_kl))
    save_fig_losses(epoch, d_losses, g_losses, r_losses_real, r_losses_fake, kl_losses, None, None,  dir_files)

    # do checkpointing
    torch.save({
        'generator': netG.state_dict(),
        'discriminator': netD.state_dict(),
        'g_optim': optimizerG.state_dict(),
        'd_optim': optimizerD.state_dict(),
        'd_losses': d_losses,
        'g_losses': g_losses,
        'r_losses_real': r_losses_real,
        'r_losses_fake': r_losses_fake,
        'kl_losses': kl_losses,
    }, dir_checkpoint+'/trained.pth')
    
    # save network after 1 learning epoch
    if epoch ==1:
            torch.save({
        'generator': netG.state_dict(),
        'discriminator': netD.state_dict(),
        }, dir_checkpoint+'/trained_epoch1.pth')

    print(f'Model successfully saved.')
