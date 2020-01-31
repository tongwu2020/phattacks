# Defending Against Physically Realizable Attacks on Image Classification

### Tong Wu, Liang Tong, Yevgeniy Vorobeychik
 

[[arXiv]](https://arxiv.org/abs/1909.09552)   **first version**


###ABSTRACT
We study the problem of defending deep neural network approaches for image classification from physically realizable attacks. First, we demonstrate that the two most scalable and effective methods for learning robust models, adversarial training with PGD attacks and randomized smoothing, exhibit very limited effec- tiveness against three of the highest profile physical attacks. Next, we propose a new abstract adversarial model, rectangular occlusion attacks, in which an ad- versary places a small adversarially crafted rectangle in an image, and develop two approaches for efficiently computing the resulting adversarial examples. Fi- nally, we demonstrate that adversarial training using our new attack yields image classification models that exhibit high robustness against the physically realizable attacks we study, offering the first effective generic defense against such attacks.


### <img src="figures/figure2.png" height="210" width="860">


## Usage


## Image Geolocation
The image geolocation is casted as an image classification problem. We use VGG16 for this task and get the image features. The code was adapted from [https://github.com/ayanc/tf-boilerplate](https://github.com/ayanc/tf-boilerplate). 

Requirement: tensorflow version 1.7.0

This section ```VGG16``` is for users who want to apply our approach on their own photo collections, where they can adopt these codes to get image feature for your image. We have provided the links and label of the images we used. 

## Album Geolocation

We provide the images features, therefore users who don't have GPUs can apply our approach directly. You need to get the score matrices for our synthetic image albums. 
```
  python albums_fea/feaGet.py -m 1 -s 16
```
```
  "-m": model number which could be 1 or 2. 
  "-s": album size which could be 16, 32, 64 or 128.
```
*Note: make sure you have run **all** the combinations of the parameter choices.* 



Contact [tongwu@wustl.edu]() with any questions. 