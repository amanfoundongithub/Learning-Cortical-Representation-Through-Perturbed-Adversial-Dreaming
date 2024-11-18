from __future__ import print_function
import os
import numpy as np
import torch.utils.data
from utils import *
from network import *


##########################################################
dir_files = "./result/cifar10/model_wnf/" 
try:
    os.makedirs(dir_files)
except OSError:
    pass

dir_checkpoint = "./checkpoint/cifar10/model_wnf"
try:
    os.makedirs(dir_checkpoint)
except OSError:
    pass

##########################################################



# TSNE setup
n_samples = 500
split = 20
n_c = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataset, unorm, img_channels = get_dataset("cifar10", "./cifar10", 32, is_train = True)
test_dataset, unorm, img_channels = get_dataset("cifar10", "./cifar10", 32, is_train = False) 

train_dataloader  = torch.utils.data.DataLoader(train_dataset, batch_size = n_samples//split, shuffle=True, drop_last=True)
train_dataloader2 = torch.utils.data.DataLoader(train_dataset, batch_size = n_samples//split, shuffle=True, drop_last=True)
train_dataloader3 = torch.utils.data.DataLoader(train_dataset, batch_size = n_samples//split, shuffle=True, drop_last=True)
train_dataloader4 = torch.utils.data.DataLoader(train_dataset, batch_size = n_samples//split, shuffle=True, drop_last=True)

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

# setup networks
netGe = Generator(ngpu = 1, nz = nz, ngf = 64, img_channels = img_channels)
netGe.apply(weights_init)
netDe = Discriminator(ngpu = 1, nz = nz, ndf = 64, img_channels = img_channels,  p_drop = 0.0)
netDe.apply(weights_init)
# send to GPU
netDe.to(device)
netGe.to(device)


## FID network
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
net_inception = InceptionV3([block_idx])
net_inception = net_inception.to(device)

if os.path.exists(dir_checkpoint+'/trained.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model (Final)...')
    checkpoint = torch.load(dir_checkpoint+'/trained.pth', map_location='cpu')
    netG.load_state_dict(checkpoint['generator'])
    netD.load_state_dict(checkpoint['discriminator'])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')
    
    
if os.path.exists(dir_checkpoint+'/trained_epoch1.pth'):
    # Load data from first epoch
    print('Loading pre-trained model (Epoch 1)...')
    checkpoint = torch.load(dir_checkpoint+'/trained_epoch1.pth', map_location='cpu')
    netGe.load_state_dict(checkpoint['generator'])
    netDe.load_state_dict(checkpoint['discriminator'])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')
    


all_inception_real = np.zeros((n_samples, 2048))
all_inception_fake = np.zeros((n_samples, 2048))

########################
# NREM before the training started 
########################
with torch.no_grad():
    for i in range(split):
        # Get real samples
        imgs, _ = next(iter(train_dataloader))
        imgs = imgs.to(device)
        imgs2, _ = next(iter(train_dataloader2))
        imgs2 = imgs2.to(device)
        
        # Latent representation
        latent_output, _ = netDe(imgs2)
        
        # NREM reconstruction (dream) 
        reconstructed_imgs2 = netGe(latent_output)
        all_inception_real[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(imgs, net_inception)
        all_inception_fake[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(reconstructed_imgs2, net_inception)
        
    frechet_dist_NREM_early = calculate_frechet(all_inception_real, all_inception_fake, net_inception)
    print("FID NREM early : "+str(frechet_dist_NREM_early))

########################
# REM before the training started 
########################
all_inception_real = np.zeros((n_samples, 2048))
all_inception_fake = np.zeros((n_samples, 2048))
with torch.no_grad():
    for i in range(split):
        imgs, _ = next(iter(train_dataloader))
        imgs = imgs.to(device)
        imgs3, _ = next(iter(train_dataloader3))
        imgs3 = imgs3.to(device)
        imgs4, _ = next(iter(train_dataloader4))
        imgs4 = imgs4.to(device)
        latent_output3, _ = netDe(imgs3)
        latent_output4, _ = netDe(imgs4)
        noise = torch.randn(latent_output3.size(), device=device)
        latent_output_dream = 0.25*latent_output3 + 0.25*latent_output4 + 0.5*noise
        rem_imgs = netGe(latent_output_dream)
        all_inception_real[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(imgs, net_inception)
        all_inception_fake[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(rem_imgs, net_inception)
    frechet_dist_REM_early = calculate_frechet(all_inception_real, all_inception_fake, net_inception)
    print("FID REM early : "+str(frechet_dist_REM_early))


########################
# NREM after the training ended 
########################
all_inception_real = np.zeros((n_samples, 2048))
all_inception_fake = np.zeros((n_samples, 2048))
with torch.no_grad():
    for i in range(split):
        imgs, _ = next(iter(train_dataloader))
        imgs = imgs.to(device)
        imgs2, _ = next(iter(train_dataloader2))
        imgs2 = imgs2.to(device)
        latent_output, _ = netD(imgs2)
        reconstructed_imgs2 = netG(latent_output)
        all_inception_real[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(imgs, net_inception)
        all_inception_fake[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(reconstructed_imgs2, net_inception)
    frechet_dist_NREM_late = calculate_frechet(all_inception_real, all_inception_fake, net_inception)
    print("FID NREM late : "+str(frechet_dist_NREM_late))

########################
# REM after the training ended 
########################
all_inception_real = np.zeros((n_samples, 2048))
all_inception_fake = np.zeros((n_samples, 2048))
with torch.no_grad():
    for i in range(split):
        imgs, _ = next(iter(train_dataloader))
        imgs = imgs.to(device)
        imgs3, _ = next(iter(train_dataloader3))
        imgs3 = imgs3.to(device)
        imgs4, _ = next(iter(train_dataloader4))
        imgs4 = imgs4.to(device)
        latent_output3, _ = netD(imgs3)
        latent_output4, _ = netD(imgs4)
        noise = torch.randn(latent_output3.size(), device=device)
        latent_output_dream = 0.25*latent_output3 + 0.25*latent_output4 + 0.5*noise
        rem_imgs = netG(latent_output_dream)
        all_inception_real[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(imgs, net_inception)
        all_inception_fake[(n_samples//split)*i:(n_samples//split)*(i+1)] = calculate_activation_statistics(rem_imgs, net_inception)
    frechet_dist_REM_late = calculate_frechet(all_inception_real, all_inception_fake, net_inception)
    print("FID REM late : "+str(frechet_dist_REM_late))


frechet_dist_NREM = [frechet_dist_NREM_early, frechet_dist_NREM_late]
frechet_dist_REM = [frechet_dist_REM_early, frechet_dist_REM_late]


try:
    os.makedirs(dir_files + "/frechet_dist")
except OSError:
    pass
torch.save({
        'frechet_dist_NREM': frechet_dist_NREM,
        'frechet_dist_REM': frechet_dist_REM,
    }, dir_files+'/frechet_dist/frechet_dist.pth')
print(f'Distances successfully saved.')
