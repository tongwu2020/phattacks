# Defending Against Physically Realizable Attacks on Image Classification

### [Tong Wu](https://tongwu2020.github.io/tongwu/), [Liang Tong](https://liang-tong.me), [Yevgeniy Vorobeychik](http://vorobeychik.com)
#### Washington University in St. Louis
 
### Paper 
[Defending Against Physically Realizable Attacks on Image Classification](https://arxiv.org/abs/1909.09552) 


## ABSTRACT

We study the problem of defending deep neural network approaches for image classification from physically realizable attacks. First, we demonstrate that the two most scalable and effective methods for learning robust models, adversarial training with PGD attacks and randomized smoothing, exhibit very limited effectiveness against three of the highest profile physical attacks. Next, we propose a new abstract adversarial model, rectangular occlusion attacks, in which an adversary places a small adversarially crafted rectangle in an image, and develop two approaches for efficiently computing the resulting adversarial examples. Finally, we demonstrate that adversarial training using our new attack yields image classification models that exhibit high robustness against the physically realizable attacks we study, offering the first effective generic defense against such attacks.

##  Motivation

A large literature has emerged on defending deep neural networks against adversarial examples on the feature space, namely l_2, l_infty etc.

However,there seem no effective methods specifically to defend against physically realizable attacks (major concern in real life).
 

## What is Physically Realizable Attack?

<img src="Figure/phattack.png" height="130" width="860">

Left three images are an example of the eyeglass frame attack. Left: original face input image. Middle: modified input image (adversarial eyeglasses superimposed on the face). Right: an image of the predicted individual with the adversarial input in the middle image. 

Right three images are an example of the stop sign attack. Left: original stop sign input image. Middle: adversarial mask. Right: stop sign image with adversarial stickers, classified as a speed limit sign.

Basically, there are three characteristics.  
1. The attack can be implemented in the physical space (e.g., modifying the stop sign);
2. the attack has low suspiciousness; this is operationalized by modifying only a small part of the object, with the modification similar to common “noise” that obtains in the real world;
3. the attack causes misclassification by state-of-the-art deep neural network


## Prepare for the experiment 
1. Clone this repository: 
```
git clone https://github.com/tongwu2020/phattacks.git
```

2. Install the dependencies:
```
conda create -n phattack
conda activate phattack
# Install following packages:
# See https://pytorch.org/ for the correct command for your system to install correct version of Pytorch 
conda install scipy pandas statsmodels matplotlib seaborn numpy 
conda install -c conda-forge opencv
# May need more packages 
```

3. Download our trained models from:

4. Run specific task:

```
cd glass 
```
or 

```
cd sign
```




Contact [tongwu@wustl.edu]() with any questions. 