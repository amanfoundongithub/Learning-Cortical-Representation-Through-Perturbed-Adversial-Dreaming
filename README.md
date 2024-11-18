
# Learning cortical representations through perturbed and adversarial dreaming

This repository contains the code to reproduce the results of the eLife publication "Learning cortical representations through perturbed and adversarial dreaming" [available  here](https://elifesciences.org/articles/76384).

## Requirements 

To install requirements:
 ```
 pip install -r requirements.txt
 
```
## Training & Evaluation 

In order to train the model execute: 
```
python PAD_Train_GAN.py
```





### Linear Classifier
In order to compute linear classification accuracy, execute: 
```
python PAD_Classify.py
```

### Linear Classifier With Occlusion
In order to compute accuracies with different level of occlusions, execute:
```
python PAD_Classify_With_Occlusion.py
```
















