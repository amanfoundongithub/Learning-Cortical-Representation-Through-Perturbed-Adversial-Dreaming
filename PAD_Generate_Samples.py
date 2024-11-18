from __future__ import print_function
import os
import torch.utils.data
import torchvision.utils as vutils
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


if os.path.exists(dir_checkpoint+'/trained.pth'):
    # Load data from last checkpoint
    print('Loading pre-trained model (Final)...')
    checkpoint = torch.load(dir_checkpoint+'/trained.pth', map_location='cpu')
    netG.load_state_dict(checkpoint['generator'])
    netD.load_state_dict(checkpoint['discriminator'])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')

# setup networks for epoch 1 
netG1 = Generator(ngpu = 1, nz = nz, ngf = 64, img_channels = img_channels)
netG1.apply(weights_init)
netD1 = Discriminator(ngpu = 1, nz = nz, ndf = 64, img_channels = img_channels,  p_drop = 0.0)
netD1.apply(weights_init)
# send to GPU
netD1.to(device)
netG1.to(device)

if os.path.exists(dir_checkpoint+'/trained_epoch1.pth'):
    # Load data from epoch 1
    print('Loading pre-trained model (Epoch 1)...')
    checkpoint = torch.load(dir_checkpoint+'/trained_epoch1.pth', map_location='cpu')
    netG1.load_state_dict(checkpoint['generator'])
    netD1.load_state_dict(checkpoint['discriminator'])
    print('Start training from loaded model...')
else:
    print('No pre-trained model detected, restart training...')

# Prepare image batches 
dataloader_iter = iter(dataloader)
image_eval1, _ = next(dataloader_iter) # first batch of images
image_eval2, _ = next(dataloader_iter) # second batch of images

image_eval1_save = image_eval1.clone()
image_eval2_save = image_eval2.clone()


# Keeps only the first 3 images of each batch 
try:
    os.makedirs(dir_files + "/eval_fig_3")
except OSError:
    pass

vutils.save_image(unorm(image_eval1_save[:3]).data, '%s/eval_fig_3/wake1_%03d.png' % (dir_files, 0), nrow=1)
vutils.save_image(unorm(image_eval2_save[:3]).data, '%s/eval_fig_3/wake2_%03d.png' % (dir_files, 0), nrow=1)


with torch.no_grad():
    
    # push to CUDA
    image_eval1 = image_eval1.to(device)
    image_eval2 = image_eval2.to(device)
    
    # Generate Latent representation of the images 
    latent_output1, _ = netD(image_eval1)
    latent_output2, _ = netD(image_eval2)
    
    # Evaluate an NREM image (generator)
    nrem = netG(latent_output1) 
    
    # Perturbe the generated images to produce a NREM image (dreamed image)
    noise = torch.randn(batch_size, nz, device=device)
    latent_rem = 0.25*latent_output1 + 0.25*latent_output2 + 0.5*noise
    
    # REM image
    rem = netG(latent_rem) 


nrem = unorm(nrem)
rem = unorm(rem)

# Reconstruct the Image
rec_image_eval1=unorm(image_eval1)
rec_image_eval2=unorm(image_eval2)

# Now we save the reconstructed images 
vutils.save_image(rec_image_eval1[:3].data, '%s/eval_fig_3/final_rec1.png' % (dir_files), nrow=1)
vutils.save_image(rec_image_eval2[:3].data, '%s/eval_fig_3/final_rec2.png' % (dir_files), nrow=1)

# Save the NREM and REM images 
vutils.save_image(nrem[:3].data, '%s/eval_fig_3/final_nrem.png' % (dir_files), nrow=1)
vutils.save_image(rem[:3].data, '%s/eval_fig_3/final_rem.png' % (dir_files), nrow=1)

with torch.no_grad():
    
    # push to CUDA
    image_eval1 = image_eval1.to(device)
    image_eval2 = image_eval2.to(device)
    
    # Generate Latent representation of the images 
    latent_output1, _ = netD1(image_eval1)
    latent_output2, _ = netD1(image_eval2)
    
    # Evaluate an NREM image (generator)
    nrem = netG1(latent_output1) 
    
    # Perturbe the generated images to produce a NREM image (dreamed image)
    noise = torch.randn(batch_size, nz, device=device)
    latent_rem = 0.25*latent_output1 + 0.25*latent_output2 + 0.5*noise
    
    # REM image
    rem = netG1(latent_rem) 


nrem = unorm(nrem)
rem = unorm(rem)

# Reconstruct the Image
rec_image_eval1=unorm(image_eval1)
rec_image_eval2=unorm(image_eval2)

# Now we save the reconstructed images 
vutils.save_image(rec_image_eval1[:3].data, '%s/eval_fig_3/ep1_rec1.png' % (dir_files), nrow=1)
vutils.save_image(rec_image_eval2[:3].data, '%s/eval_fig_3/ep1_rec2.png' % (dir_files), nrow=1)

# Save the NREM and REM images 
vutils.save_image(nrem[:3].data, '%s/eval_fig_3/ep1_nrem.png' % (dir_files), nrow=1)
vutils.save_image(rem[:3].data, '%s/eval_fig_3/ep1_rem.png' % (dir_files), nrow=1)





