# Defending Against Physically Realizable Attacks on Image Classification

### [Tong Wu](https://tongwu2020.github.io/tongwu/), [Liang Tong](https://liang-tong.me), Yevgeniy Vorobeychik(http://vorobeychik.com)

#### [Department of Computer Science and Engineering](https://cse.wustl.edu/Pages/default.aspx)
#### Washington University in St. Louis
 

[Defending Against Physically Realizable Attacks on Image Classification](https://arxiv.org/abs/1909.09552)   **first version**


## ABSTRACT

We study the problem of defending deep neural network approaches for image classification from physically realizable attacks. First, we demonstrate that the two most scalable and effective methods for learning robust models, adversarial training with PGD attacks and randomized smoothing, exhibit very limited effec- tiveness against three of the highest profile physical attacks. Next, we propose a new abstract adversarial model, rectangular occlusion attacks, in which an ad- versary places a small adversarially crafted rectangle in an image, and develop two approaches for efficiently computing the resulting adversarial examples. Fi- nally, we demonstrate that adversarial training using our new attack yields image classification models that exhibit high robustness against the physically realizable attacks we study, offering the first effective generic defense against such attacks.

<img src="Figure/phattack.png" height="130" width="860">

Left three images are an example of the eyeglass frame attack. Left: original face input image. Middle: modified input image (adversarial eyeglasses superimposed on the face). Right: an image of the predicted individual with the adversarial input in the middle image. 

Right three images are an example of the stop sign attack. Left: original stop sign input image. Middle: adversarial mask. Right: stop sign image with adversarial stickers, classified as a speed limit sign.


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