
# Defending Against Physically Realizable Attacks on Image Classification

In Proceedings of the 8th International Conference on Learning Representations (ICLR’20)

### [Tong Wu](https://tongwu2020.github.io/tongwu/), [Liang Tong](https://liang-tong.me), [Yevgeniy Vorobeychik](http://vorobeychik.com)
#### Washington University in St. Louis
 
### Paper 
[Defending Against Physically Realizable Attacks on Image Classification](https://arxiv.org/abs/1909.09552) 


## Abstract

We study the problem of defending deep neural network approaches for image classification from physically realizable attacks. First, we demonstrate that the two most scalable and effective methods for learning robust models, adversarial training with PGD attacks and randomized smoothing, exhibit very limited effectiveness against three of the highest profile physical attacks. Next, we propose a new abstract adversarial model, rectangular occlusion attacks, in which an adversary places a small adversarially crafted rectangle in an image, and develop two approaches for efficiently computing the resulting adversarial examples. Finally, we demonstrate that adversarial training using our new attack yields image classification models that exhibit high robustness against the physically realizable attacks we study, offering the first effective generic defense against such attacks.

##  Motivation

A large literature has emerged on defending deep neural networks against adversarial examples on the feature space, namely l_2, l_infty etc. However,there seems no effective methods specifically to defend against physically realizable attacks (major concern in real life).
 

## What is Physically Realizable Attack?

<img src="Figure/phattack.png" height="130" width="860">

(a)Left three images are an example of the eyeglass frame attack. Left: original face input image. Middle: modified input image (adversarial eyeglasses superimposed on the face). Right: an image of the predicted individual with the adversarial input in the middle image. 

(b)Right three images are an example of the stop sign attack. Left: original stop sign input image. Middle: adversarial mask. Right: stop sign image with adversarial stickers, classified as a speed limit sign.

Basically, there are three characteristics.  
1. The attack can be implemented in the physical space (e.g., modifying the stop sign);
2. the attack has low suspiciousness; this is operationalized by modifying only a small part of the object, with the modification similar to common “noise” that obtains in the real world;
3. the attack causes misclassification by state-of-the-art deep neural network

## Abstract Attack Model: Rectangular Occlusion Attacks (ROA)

We introduce a rectangle which can be placed by the adversary anywhere in the image. Then the attacker can furthermore introduce l_infty noise inside the rectangle with epsilon = 255.

#### How to determine the Location of this rectangle

1. Exhaustive Searching : Adding a grey rectangular sticker to image, considering all possible locations and choosing the worst-case attack
2. Gradient Based Searching : Computing the magnitude of the gradient w.r.t each pixel, considering all possible locations and choosing C locations with largest magnitude. Exhaustively searching among these C locations.

#### Examples 

<img src="Figure/ROA.png" height="130" width="860">

Examples of the ROA attack on face recognition, using a rectangle of size 100 × 50. 

(a) Left three images. Left: the original A. J. Buckley’s image. Middle: modified input image (ROA superimposed on the face). Right: an image of the predicted individual who is Aaron Tveit with the adversarial input in the middle image. 

(b) Right three images. Left: the original Abigail Spencer’s image. Middle: modified input image (ROA superimposed on the face). Right: an image of the predicted individual who is Aaron Yoo with the adversarial input in the middle image.

## Defense against Occlusion Attacks (DOA) & Results 

We apply the adversarial training approach for ROA to fine tune the clean model, achieving significant improvement compared to conventional robust classifers. 

<div>
<img style="float:left" src="Figure/Main_glassAll.png" height="288" width="396" />
<img style="float:right" src="Figure/Comshapeexh_glassAtt.png" height="288" width="396" />
</div>


Effectiveness of DOA on face recognition against eyeglass frame attacks.

(a) Left image: Performance of DOA (using the 100 * 50 rectangle) against the eyeglass frame attack in
comparison with conventional methods. Comparison between DOA, adversarial training, and
randomized smoothing (using the most robust variants of these).

(b) Right image:Comparing DOA
performance for different rectangle dimensions and numbers of PGD iterations inside the rectangle.

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
conda install scipy pandas statsmodels matplotlib seaborn numpy 
conda install -c conda-forge opencv
pip install foolbox==2.3.0
```
See [Pytorch](https://pytorch.org/) for the command for your system to install correct version of Pytorch 
May need more packages

3. Run specific task: for Face Recognition, 

```
cd glass 
```
or for traffic sign classification, 

```
cd sign
```

## Try DOA in your own dataset 
```
cd ROA
```
 <p> View the Quick Demo in Google Colab <a href="https://colab.research.google.com/gist/tongwu2020/bbf836348be405f3bebe96d0ea12df43/roa-doa_test.ipynb" style="width:50px;height:60px;" >
      <img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" style="width:50px;height:60px;">
 
 </a>

</p>


Contact [tongwu@wustl.edu]() with any questions. 
