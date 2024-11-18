from __future__ import print_function
import os
import numpy as np
import torch.optim as optim
import torch.utils.data
from utils import *
from network import *


##########################################################

n_epochs_c =40
lrC = 2e-4
drop_rate = 40


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

n_train = 50000
n_test = 10000

# get train dataset
dataset, unorm, img_channels = get_dataset("cifar10", "./cifar10", 32, drop_rate = drop_rate/100)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size = n_train, shuffle=True, drop_last=True)

# get test dataset
test_dataset, unorm, img_channels = get_dataset("cifar10", "./cifar10", 32, is_train = False, drop_rate = drop_rate/100) 
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = n_test, shuffle=False, drop_last=True)

# some hyper parameters
nz = 256
batch_size = 64
num_classes = 10

# setup networks
netG = Generator(ngpu = 1, nz = nz, ngf = 64, img_channels = img_channels)
netG.apply(weights_init)
netD = Discriminator(ngpu = 1, nz = nz, ndf = 64, img_channels = img_channels,  p_drop = drop_rate/100)
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



train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []


# project all train images on latent space
print("Storing training representations ...")
image, label = next(iter(train_dataloader))
image, label = image.to(device), label.to(device)
netD.eval()
with torch.no_grad():
    latent_output, _ = netD(image)
    train_features = latent_output.cpu()
    train_labels = label.cpu().long()

# project all test images on latent space
print("Storing validation representations ...")
image, label = next(iter(test_dataloader))
image, label = image.to(device), label.to(device)
netD.eval()
with torch.no_grad():
    latent_output, _ = netD(image)
    test_features = latent_output.cpu()
    test_labels = label.cpu().long()

# create new datasets of latent vectors
linear_train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
linear_test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)


# create new dataloaders of latent vectors
linear_train_loader = torch.utils.data.DataLoader(linear_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
linear_test_loader = torch.utils.data.DataLoader(linear_test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


# loss function for linear classifier
class_criterion = nn.CrossEntropyLoss()

# initialize linear classifier
classifier = OutputClassifier(nz, num_classes=num_classes)
classifier.to(device)
optimizerC = optim.SGD(classifier.parameters(), lr = lrC)

for epoch in range(n_epochs_c):

    store_train_acc = []
    store_test_acc = []
    store_train_loss = []
    store_test_loss = []

    print("Training on train set...")

    for feature, label in linear_train_loader:
        feature, label = feature.to(device), label.to(device)
        classifier.train()
        optimizerC.zero_grad()
        class_output = classifier(feature)
        # compute classif error
        class_err = class_criterion(class_output, label)
        # update classifier with backprop
        class_err.backward()
        optimizerC.step()
        # store train metrics
        train_acc = compute_acc(class_output, label)
        store_train_acc.append(train_acc)
        store_train_loss.append(class_err.item())


    print("testing on test set...")
    # compute test accuracy
    for feature, label in linear_test_loader:
        feature, label = feature.to(device), label.to(device)
        classifier.eval()
        class_output = classifier(feature)
        class_err = class_criterion(class_output, label)
        # store test metrics
        test_acc = compute_acc(class_output, label)
        store_test_acc.append(test_acc)
        store_test_loss.append(class_err.item())

    print('[%d/%d]  train_loss: %.4f  test_loss: %.4f  train_acc: %.4f  test_acc: %.4f'
              % (epoch, n_epochs_c, np.mean(store_train_loss), np.mean(store_test_loss), np.mean(store_train_acc), np.mean(store_test_acc)))



    torch.save({
    'classifier': classifier.state_dict(),
    }, dir_checkpoint + '/trained_classifier.pth')
    print(f'Classifier successfully saved.')


    # train average metrics
    train_accuracies.append(np.mean(store_train_acc))
    train_losses.append(np.mean(store_train_loss))
    # test average metrics
    test_accuracies.append(np.mean(store_test_acc))
    test_losses.append(np.mean(store_test_loss))
    
    torch.save({
    'train_accuracies': train_accuracies,
    'test_accuracies': test_accuracies,
    'train_losses': train_losses,
    'test_losses': test_losses,
    }, dir_files+'/classifier/occlusion_accuracies_' + str(drop_rate) + '.pth')
    print(f'Accuracies successfully saved.')

# do checkpointing
try:
    os.makedirs(dir_files + "/classifier")
except OSError:
    pass

# Plot loss and accuracies during training
e = np.arange(0, len(train_accuracies))
fig = plt.figure(figsize=(12, 6))

# Loss plot
ax1 = fig.add_subplot(121)
ax1.plot(e, train_losses, label='Train Loss', color='blue', marker='o', markersize=4, linestyle='-', linewidth=2)
ax1.plot(e, test_losses, label='Test Loss', color='orange', marker='x', markersize=4, linestyle='--', linewidth=2)
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Loss', fontsize=14)
ax1.set_title('Losses During Training', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=12)

# Accuracy plot
ax2 = fig.add_subplot(122)
ax2.plot(e, train_accuracies, label='Train Accuracy', color='green', marker='o', markersize=4, linestyle='-', linewidth=2)
ax2.plot(e, test_accuracies, label='Test Accuracy', color='red', marker='x', markersize=4, linestyle='--', linewidth=2)
ax2.set_ylim(0, 100)
ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Accuracy (%)', fontsize=14)
ax2.set_title('Accuracy During Training', fontsize=16)
ax2.legend(fontsize=12)
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=12)

# Save the figure
plt.tight_layout()  # Adjust the layout
fig.savefig(dir_files + '/classifier/linear_classif_occlusion_' + str(drop_rate) + '.png', bbox_inches='tight')
