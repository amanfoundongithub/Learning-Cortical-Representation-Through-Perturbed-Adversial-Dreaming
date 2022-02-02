
# Learning cortical representations through perturbed and adversarial dreaming

This repository contains the code to reproduce the results of the eLife submission "Learning cortical representations through perturbed and adversarial dreaming" (also available on [ArXiv](https://arxiv.org/abs/2109.04261)).

## Requirements 

To install requirements:
 ```
 pip install -r requirements.txt
 
```
## Training & Evaluation 

In order to train the model with for example the CIFAR-10 dataset, for 50 epochs, with all phases (Wake, NREM REM), execute: 
```
python main_PAD.py --dataset 'cifar10' --niter 50 --batchSize 64  --outf 'model_wnr' --nz 256  --is_continue 1 --W 1.0  --N 1.0  --R 1.0 
```

Setting one of the phase parameters (```W, N, R```) to zero will remove the phase from training. At each epoch, the endoder and generator networks, as well as the training losses, are saved into the file ```trained.pth```. In order to compute linear classification accuracy, execute: 
```
python linear_classif.py --dataset 'cifar10' --niterC 20  --outf 'model_wnr' --nz 256
```

Classification accuracy can be computed at each epoch by running the following command in a job file:
```
dset='cifar10'
folder='model_wnr'
for i in {1..50}
do
        python main_PAD.py --dataset $dset --niter $i --batchSize 64  --outf $folder --nz 256  --is_continue 1 --W 1.0  --N 1.0  --R 1.0  --epsilon 0.0 
        python linear_classif.py --dataset $dset --niterC 20  --outf $folder --nz 256
done
```



