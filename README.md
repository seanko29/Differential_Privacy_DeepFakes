# Differential Privacy DeepFakes
Official Repository for "Analysis of Obfuscation of Deepfake Images in Differential Privacy Settings" published in KAIA2022.

#### Prerequisites

+ python >= 3.7.0
+ PyTorch >= 1.8.0
+ CUDA 11.4 (on RTX A5000 x 4)

```python
pip install -r requirements.txt
```
-------------
#### Applying differential privacy to images
In this repository, we apply three differential privacy methods to DeepFake images named pixelation, SVD and snow. For each method, we provide 3 jupyter notebook files as follows:

+ `differential privacy pixelation.ipynb`

+ `differential privacy svd.ipynb`

+ `differential privacy snow.ipynb`

In each notebook, there is a header where it states "convert deepfake images to DP". 
By changing the directory of the deepfake image and directory for saving images, one may run the code below in each DP notebook.

```python
import os
df_imgdir = glob('/home/data/deepfake_privacy/FF_original/FaceForensics++/Face2Face/test/real/*/')
fake_img_dir = '/home/data/deepfake_privacy/FF_original/FaceForensics++/Face2Face/test/real/'
k = 7
epsilon = 0.01

for idx, folder in enumerate(df_imgdir):
    
    # extract the last part of the folder for joining path
    imgfolder = folder.split('/')[-2]
    if not os.path.exists('/home/data/deepfake_privacy/ff_priv/Face2Face/pixel/test/real/{0}'.format(imgfolder)):
        os.makedirs(os.path.join('/home/data/deepfake_privacy/ff_priv/Face2Face/pixel/test/real/', imgfolder))
    subfolder = os.path.join('/home/data/deepfake_privacy/ff_priv/Face2Face/pixel/test/real/', imgfolder)
    
    # subfolder = location of saved area
    # globfolder = location of train data
    globfolder = fake_img_dir + imgfolder + '/*'
    
    print(globfolder, 'glob folder')

    # generating pixelation image per every image of each subfolder
    for idx2, img in enumerate(glob(globfolder)):
        image = Image.open(img)
        a = pillow_to_numpy(image)
        dpimg = dp_pixelate_images_singleimg(a, target_h, target_w, m, eps)
        dpimg = dpimg.astype('uint8')
    

        cv2.imwrite(subfolder+'/folder_{0}_pixel_test_{1}.jpg'.format(idx, idx2),cv2.cvtColor(dpimg, cv2.COLOR_RGB2BGR))
        if idx2 == 10:
            print(subfolder, 'sub folder')
            
print('######finished conversion!!##########')

```

-------------
#### Running Train Code
Adjust the hyperparameters in `train.py` or customize your own in `train.sh`
```python
python tools/train.py
```
```python
sh tools/train.sh
```

#### Running Test Code
```python
sh tools/test.sh
```

The code for training/testing works in data parallel. You may adjust the number of GPUs to use. In this work, we used upto 4 GPUs in training and testing.

