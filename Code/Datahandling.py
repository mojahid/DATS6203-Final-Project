
## imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import glob
from shutil import copy
import matplotlib.pyplot as plt
import os, os.path


## The data comes in two directories:
##  1- Original
##  2- Photoshopped
## Each image under the original directory has a corresponing one more photo shopped version
## This one or more photoshopped images are placed under the photoshopped directory under a
## subdirectory having the same original image name
##  original---
##             |---Image1.jpg
##
##  photoshopped---
##             |---Image1
##                     |--Image1_0.jpg
##                     |--Image1_1.jpg

## problem with the data:
## 1- Size ranges between 10K and 13mb
## 2- Few images has 0 or 1 kb size which is empty
##

BASE_PATH = '/home/ubuntu/MLP/DATS6203-Final-Project/Data/'

def get_photoshopped_images(image_name):
    """ This function retrieve photoshopped images that corresponds to an original image.
    The function returns a list of images in an array
    Keyword arguments:
    image_name -- name of the original image without extension
    """
    img_data = []
    original_img_path = BASE_PATH + 'originals/'+ image_name +'.jpg'
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (300, 300))
    img_data.append(img)
    path = BASE_PATH + 'photoshops'+ os.path.sep + image_name
    for file in os.listdir(path):
        if file[-4:] == '.jpg':
            filename = path + os.path.sep + file
            img_data.append(cv2.resize(cv2.imread(filename), (200, 200)))

    return img_data

def plot_image_set(img_data):
    """ This function display a plot of  original and its corresponding photoshopped images provided in the imag_data.
    Original image is the first image in the array
    Keyword arguments:
    img_data -- List of images coming get_photoshopped_images()
    """
    frame_rgb = []
    f, axarr = plt.subplots(int(np.ceil(len(img_data)/2)), 2, figsize=(8, 8))
    switch = 0
    for i in range(int(np.ceil(len(img_data)/2))):
        if((len(img_data) % 2 != 0) and (i == np.ceil(len(img_data)/2) -1)):
            b, g, r = cv2.split(img_data[i*2 + 0])
            frame_rgb.append(cv2.merge((r, g, b)))
            axarr[i, 0].imshow(frame_rgb[i*2 + 0])
            f.delaxes(axarr[i, 1])
        else:
            for j in range(2):
                index = i*2
                b, g, r = cv2.split(img_data[index+j])
                frame_rgb.append(cv2.merge((r, g, b)))
                axarr[i, j].imshow(frame_rgb[index+j])

    plt.show()
    return

def update_image_directories():
    dir_src = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/photoshops/"
    dir_dst = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/DataClasses/fake/"
    original_count = 0
    fake_count = 0
    for file in glob.iglob('%s/**/*.*' % dir_src, recursive=True):
        copy(file, dir_dst)
        fake_count = fake_count + 1
    dir_src = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/originals/"
    dir_dst = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/DataClasses/original/"
    for file in glob.iglob('%s/**/*.*' % dir_src, recursive=True):
        copy(file, dir_dst)
        original_count = original_count + 1
    return original_count, fake_count


def target_process():
    img_name_fake = []
    img_path_fake = BASE_PATH + 'DataClasses' + os.path.sep + 'fake'
    img_path_fake = glob.glob(os.path.join(img_path_fake, "*.jpg"))
    for img in img_path_fake:
        img = img.replace('/', '')
        img_name_fake.append([img[54:],0])

    df_fake = pd.DataFrame(img_name_fake, columns=['image_name', 'target'])

    img_name_original = []
    img_path_original = BASE_PATH + 'DataClasses' + os.path.sep + 'original'
    img_path_original = glob.glob(os.path.join(img_path_original, "*.jpg"))
    for img in img_path_original:
        img = img.replace('/', '')
        img_name_original.append([img[58:], 1])

    df_original = pd.DataFrame(img_name_original, columns=['image_name', 'target'])
    df_list =[df_fake,df_original]
    df = pd.concat(df_list)
    # shuffle the DataFrame rows
    df = df.sample(frac=1)

    return (df)

#image_name = '141vnd'
#image_name = '100d24'
#image_name = '1085it'

##img_data = get_photoshopped_images(image_name)
##plot_image_set(img_data)
#original_count, fake_count = update_image_directories()
#print(original_count)
#print(fake_count)
#simple version for working with CWD
## Under the photoshopped directory, there are thousands of subdirectories that corr

print(create_excel_image())
