## imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
from tqdm import tqdm
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
from random import shuffle
from sklearn.model_selection import train_test_split
import glob
from shutil import copy
import matplotlib.pyplot as plt
import os, os.path
import Image_Augmentation

TRAIN_DIR ='train'
TEST_DIR ='test1'
N_EPOCH = 10
IMG_SIZE =50
LR= 1e-3
MODEL_NAME= 'fake_img_classification'

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

BASE_PATH = '/home/ubuntu/DATS6203-Final-Project/Data/'

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

    """ This function moves images from their original extracted location to the new location
    without any sub folders in the photoshopped directory. It also ignores images below 12KB
    Keyword arguments:
    img_data -- List of images coming get_photoshopped_images()
    """

def update_image_directories():
    dir_src = r"/home/ubuntu/DATS6203-Final-Project/Data/photoshops/"
    dir_dst = r"/home/ubuntu/DATS6203-Final-Project/Data/Classes01/photoshopped/"
    original_count = 0
    fake_count = 0
    for file in glob.iglob('%s/**/*.*' % dir_src, recursive=True):
        op1 = np.random.randint(1, 8)
        if (os.path.getsize(file) > 12*1024) and (cv2.imread(file) is not None) and (op1 == 1):
            copy(file, dir_dst)
            fake_count = fake_count + 1
    dir_src = r"/home/ubuntu/DATS6203-Final-Project/Data/originals/"
    dir_dst = r"/home/ubuntu/DATS6203-Final-Project/Data/Classes01/original/"
    for file in glob.iglob('%s/**/*.*' % dir_src, recursive=True):
        if (os.path.getsize(file) > 12 * 1024) and (cv2.imread(file) is not None):
            copy(file, dir_dst)
            original_count = original_count + 1
    return original_count, fake_count

def augmentation_resampling():
    dir_src = r"/home/ubuntu/DATS6203-Final-Project/Data/Classes01/original/"
    for file in glob.iglob('%s/**/*.*' % dir_src, recursive=True):
        print(file)
        img = Image_Augmentation.flip_crop_image(cv2.imread(file), -1)
        new_name = os.path.splitext(file)[0]+'_1.jpg'
        cv2.imwrite(new_name, img)
        img = Image_Augmentation.flip_crop_image(cv2.imread(file), 0)
        new_name = os.path.splitext(file)[0]+'_2.jpg'
        cv2.imwrite(new_name, img)
        img = Image_Augmentation.flip_crop_image(cv2.imread(file), 1)
        new_name = os.path.splitext(file)[0]+'_3.jpg'
        cv2.imwrite(new_name, img)
        img = Image_Augmentation.rotate_image(cv2.imread(file))
        new_name = os.path.splitext(file)[0]+'_4.jpg'
        cv2.imwrite(new_name, img)
    return

def create_noise_analysis(img):
    img2 = cv2.medianBlur(img, 3)
    return img - img2

def create_ela(img):
    cv2.imwrite("temp.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img2 = cv2.imread("temp.jpg")
    cv2.imwrite("temp.jpg", img2, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img2 = cv2.imread("temp.jpg")
    diff = 15 * cv2.absdiff(img, img2)
    return diff

def process_filter_on_data(filter,directory_name):
    dir_src_1 = r"/home/ubuntu/DATS6203-Final-Project/Data/Classes01/original/"
    dir_dst_1 = r"/home/ubuntu/DATS6203-Final-Project/Data" + directory_name +"/original/"

    dir_src_2 = r"/home/ubuntu/DATS6203-Final-Project/Data/Classes01/photoshopped/"
    dir_dst_2 = r"/home/ubuntu/DATS6203-Final-Project/Data/" + directory_name + "/photoshopped/"
    for file in glob.iglob('%s/**/*.*' % dir_src_1, recursive=True):
        filename = os.path.basename(file)
        filename = dir_dst_1 + filename
        if filter == 'ela':
            cv2.imwrite(filename,create_ela(cv2.imread(file)))
        elif filter == 'noise':
            cv2.imwrite(filename, create_noise_analysis(cv2.imread(file)))

    for file in glob.iglob('%s/**/*.*' % dir_src_2, recursive=True):
        filename = os.path.basename(file)
        filename = dir_dst_2 + filename
        if filter == 'ela':
            cv2.imwrite(filename,create_ela(cv2.imread(file)))
        elif filter == 'noise':
            cv2.imwrite(filename, create_noise_analysis(cv2.imread(file)))

#image_name = '141vnd'
image_name = '10j990'
#image_name = '1085it'

img_data = get_photoshopped_images(image_name)


#augmentation_resampling()
plot_image_set(img_data)
#original_count, fake_count = update_image_directories()
#print(original_count)
#print(fake_count)
# simple version for working with CWD
