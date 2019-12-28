# Defending Against Physically Realizable Attacks on Image Classification
 

[[arXiv]](https://arxiv.org/abs/1909.09552)   **first version**



A person's physical whereabouts is sensitve information that many desire to keep private. We consider the specific issue of location privacy as potentially revealed by posting photo collections by carefully pruning select photos from photo collections before these are posted publicly.  
<img src="figures/figure2.png" height="210" width="860">

 
## Usage
Clone our repo and create a sub-directory named ```dataset``` under the home directory. 
Download our datasets from ```release``` and put them under ```geoPrivacyAlbum/dataset```. Read the document of the dataset in the ```release```. Please read the ```Album Geolocation``` section and run the code to get the score matrices for synthetic albums before running ```Protection``` strategies.  

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


## Protection Strategies - Synthetic Albums
We cosider the problem of censoring photo collections in order to preserve geolocation privacy: selecting a minimal set of images to delete from a given collection, such that the ture location is not included in the most likely location predicted from the remaning images. 
### IP
1. Top-k Guarantee: the first variant seeks to minimize the number of deletions, while ensuring that the resulting scores from the censored image set push the true location out of the top-k location results.  

```
  python syntheticAlbums/IP/topkProtect.py -k 5 -s 16
```
```
  "-k": could be any integer less than 512, where we use 1 and 5 in the paper. 
  "-s": album size which could be 16, 32, 64 or 128.
```

2. Fixed Deletion Budget: another variant works with a fixed budget on the number of images to delete, while trying to maiximize the rank of the true lable in the reuslting score vector. 

```
  python syntheticAlbums/IP/delFixed.py -p 4 -s 16
``` 
```
  "-p": deletion budget which could be any integer less the album size.
```


3. User Preference: users may want to specify a set of images that should not be deleted. Specifically, in this show case, users want to keep i<sup>th</sup> photo ranking by the score. 
```
  python syntheticAlbums/IP/user.py -k 5 -s 16
```


4. Black-box: this setting is used to evaluate the case when we do not have access to true classifier scores but only a proxy. We train two VGG network models, model1 and model2, each on two separate halves of the training set. 
```
  python syntheticAlbums/IP/blackbox.py -k 5 -s 16 -e 1.54 
```
```
  "-e": robustness level. You can use some float number between 0 and 4. 
```

5. Different budgets: the optimal solution may involve keeping some of the photos that would have been deleted under a lower budget.(result shown in table 4.) 

```
  python syntheticAlbums/IP/diffBudget.py -p1 4 -p2 8 -s 16
```
```
  "-p1" and "p2": two deletion budgets to be compared. 
```
### Heuristic 
1. Top-k Guarantee

```
  python syntheticAlbums/Heuristic/topkProtect.py -k 5 -s 16
```
```
  "-k": could be any integer less than 512, where we use 1 and 5 in the paper. 
  "-s": album size which could be 16, 32, 64 or 128.
```

2. Fixed Deletion Budget

```
  python syntheticAlbums/Heuristic/delFixed.py -p 4 -s 16
``` 
```
  "-p": deletion budget which could be any integer less the album size.
```



## Protection Strategies - Flickr
Get the protection strategies for Flickr photo collections with various number of photos.

### IP

1. Top-k Guarantee
```
  python Flickr/IP/topkProtect.py -k 5 
```

2. Fixed Deletion Budget
```
  python Flickr/IP/delFixed.py -p 0.25
```
```
  "-p": the ratio of deletion budget. The deletion budget will be the product of this input ratio and the album size. 
```
3. User Preference
```
  python Flickr/IP/delFixed.py -k 5
```

### Heuristic

1. Top-k Gaarantee
```
  python Flickr/Heuristic/topkProtect.py -k 5 
```

2. Fixed Deletion Budget
```
  python Flickr/Heuristic/delFixed.py -p 0.25
```
 


Contact [jinghan.yang@wustl.edu]() with any questions. 