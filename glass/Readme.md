# Face Recognition 

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



### 2. Adversarial Training & Curriculum Adversarial Training

#### L_inf adversarial training 
```
python linf_retrain.py {}.pt  -eps 4 -alpha 1 -iters 20 -out 70 -epochs 30
```
{} name of your model want to retrain (only for curriculum adversarial training), if doing adversarial training, fill in anything you want to run.

eps is the epsilon of l_infty bound;
alpha is the learning rate of PGD;
iters is the iterations of PGD; 
out is name of your output models; 
epochs is epochs you want to train; 

Currently, the model is used to adversarial training. For curriculum adversarial training, 
change the code in ```if __name__ == "__main__":``` refer to roughly line 131. 


### 3. Randomized Smoothing 

#### Training the model with Guassian noise
```
python linf_retrain.py {}.pt -out 70 -sigma 1
```
{} name of your model want to train from a clean model, right now it is training from pretrain weight,
so type anything to fill in {}. out is name of your output models (perfer a int); sigma is the sigma of gaussian noise add to original images.

### 4. Defending against Rectangular Occlusion Attacks
```
python sticker_retrain.py {}.pt -alpha 4 -iters 50 -out 99 -search 1 -epochs 5
```
{}.pt is the name of original model you want to train with DOA, you can use our model (new_ori_model.pt);
alpha is learning rate of PGD;
iters is the iterations of PGD;
out is name of your final model;
search is method of searching, '0' is exhaustive_search, '1' is gradient_based_search";
epochs is the epoch you want to fine tune your network;


#### Testing a model with original images
```
python python origin_test {}.pt
```
where {} is the name of your model want to test




