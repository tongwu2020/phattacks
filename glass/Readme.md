# Face Recognition 

## Prepare for the experiment 

Download the [Data](https://github.com/tongwu2020/phattacks/releases/tag/Data%26Model) and put into 'glass' file 

Go to experiment folder to run our experiment.
```
cd experiment 
```
## Training Models
### 1. Transfer learning the VGGFace with original images 
Download the [VGGFace Pretrained Model](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz) (About 540M)

Unzip it and put it in this file (glass) 
```
python origin_train.py
```
You don't have to download the VGGFace Pretrained Model, since we released a transfered model (About 150M) that we used.
See [Models](https://github.com/tongwu2020/phattacks/releases/tag/Data%26Model)  to find a trained model


### 2. Adversarial Training & Curriculum Adversarial Training

#### L_inf adversarial training 
Download the [VGGFace Pretrained Model](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz) (About 540M) as well.
```
python linf_retrain.py {}.pt  -eps 4 -alpha 1 -iters 20 -out 70 -epochs 30
```
(a){} name of your model want to retrain (only for curriculum adversarial training), if doing adversarial training, fill in anything you want to run.
(b)'eps' is the epsilon of l_infty bound;
(c)'alpha' is the learning rate of PGD;
(d)'iters' is the iterations of PGD; 
(e)'out' is name of your output models; 
(f)'epochs' is epochs you want to train; 

Currently, the model is used to adversarial training. For curriculum adversarial training, 
change the code in ```if __name__ == "__main__":``` refer to roughly line 131. 

You don't have to download the VGGFace Pretrained Model, since we released a L_inf model (About 150M) that we used.
See [Models](https://github.com/tongwu2020/phattacks/releases/tag/Data%26Model) to find a trained model



### 3. Randomized Smoothing 

#### Training the model with Guassian noise
```
python linf_retrain.py {}.pt -out 70 -sigma 1
```
(a){} name of your model want to train from a clean model, right now it is training from pretrain weight,
so type anything to fill in {}.
(b)'out' is name of your output models (perfer a int); 
(c)'sigma' is the sigma of gaussian noise add to original images.

See [Models](https://github.com/tongwu2020/phattacks/releases/tag/Data%26Model) to find a trained model

### 4. Defending against Rectangular Occlusion Attacks
```
python sticker_retrain.py {}.pt -alpha 4 -iters 50 -out 99 -search 1 -epochs 5
```
(a){}.pt is the name of original model you want to train with DOA, you can use our model (new_ori_model.pt);
(b)'alpha' is learning rate of PGD;
(c)'iters' is the iterations of PGD;
(d)'out' is name of your final model;
(e)'search' is method of searching, '0' is exhaustive_search, '1' is gradient_based_search";
(f)'epochs' is the epoch you want to fine tune your network;

See [Models](https://github.com/tongwu2020/phattacks/releases/tag/Data%26Model) to find a trained model

## Testing Models

### 1.Testing a model with original images
```
python python origin_test {}.pt
```
where {} is the name of your model want to test


### 2.Testing a model against L_inf & L_2 attacks

Test the L_inf robustness for single model
```
python linf_attack.py {}.pt
```

Test the L_2 robustness for single model
```
python l2_attack.py {}.pt
```
where {} is the name of your model want to test

Note that you cannot attack randomized smoothing using those following commands, please use smooth_l2attack.py

Test the L_2 robustness for randomized smoothing 
```
python smooth_l2attack.py {}.pt -sigma 1 -outfile output1
```
{}.pt is the name of gaussian model you need to train by gaussian_train.py;
'sigma' is the sigma of gaussian noise (I use same sigma with the sigma training the gaussian model);
'outfile' is the file name of your output file;

### 3.Testing a model against eyeglassframe attacks

The attack is in digit space (not involved rotation and scale) (fixed eyeglass frame mask),
and untargeted (maximize the loss of (f(x),y) )

Test the robustness against eyeglassframe attacks for single model
```
python linf_attack.py {}.pt
```
{} is the name of your model want to attack. Note that you cannot attack randomized smoothing in this file, 
please use smooth_glassattack.py;
we have default iterations of attacks that used in experiment, which is (1, 2, 3, 5, 7, 10, 20, 50, 100, 300) 

Test the robustness against eyeglassframe attacks for randomized smoothing
```
python smooth_glassattack.py {}.pt -sigma 1 -outfile output1
```
{}.pt is the name of gaussian model you need to train by gaussian_train.py;
'sigma' is the sigma of gaussian noise (I use same sigma with the sigma training the gaussian model)
'outfile' is the file name of your output file;we have default iterations of attacks that used in experiment, which is (1, 2, 3, 5, 7, 10, 20, 50, 100, 300)


### 4.Testing a model against adversarial patch

Test the robustness against adversarial patch for single model
```
python make_patch.py {}.pt
```
{} is the name of your model want to attack.
Note that you cannot attack randomized smoothing in this file, please use smooth_patch.py

Test the robustness against adversarial patch for randomized smoothing
```
python smooth_patch.py {}.pt -sigma 1
```
{} is the name of your model want to attack.
'sigma' is the sigma of randomized smoothing 

### 5.Testing a single model against JSMA 

```
python JSMA.py {}.pt
```
{}.pt is the name of model you want to attack by JSMA

Note that the output will be accuracy when changing 10,100,1000,10000 points
refer to line 125, print count to calculate the exact curve 

### 6.Testing a single model against other physical attacks 

```
python strange_retrain.py {}.pt
```
{}.pt is the name of your model want to attack


