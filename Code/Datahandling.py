
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
from sklearn.model_selection import train_test_split
import glob
from shutil import copy
import matplotlib.pyplot as plt
import os, os.path
import Image_Augmentation


TRAIN_DIR ='train'
TEST_DIR ='test1'
N_EPOCH = 10
IMG_SIZE =200
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

    """ This function moves images from their original extracted location to the new location
    without any sub folders in the photoshopped directory. It also ignores images below 12KB
    Keyword arguments:
    img_data -- List of images coming get_photoshopped_images()
    """
def update_image_directories():
    dir_src = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/photoshops/"
    dir_dst = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/DataClasses/Classes01/photoshopped/"
    original_count = 0
    fake_count = 0
    for file in glob.iglob('%s/**/*.*' % dir_src, recursive=True):
        if (os.path.getsize(file) > 12*1024) and (cv2.imread(file) is not None):
            copy(file, dir_dst)
            fake_count = fake_count + 1
    dir_src = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/originals/"
    dir_dst = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/DataClasses/Classes01/original/"
    for file in glob.iglob('%s/**/*.*' % dir_src, recursive=True):
        if (os.path.getsize(file) > 12 * 1024) and (cv2.imread(file) is not None):
            copy(file, dir_dst)
            original_count = original_count + 1
    return original_count, fake_count

def augmentation_resampling():
    dir_src = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/DataClasses/Classes01/original/"
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
    dir_src_1 = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/DataClasses/Classes01/original/"
    dir_dst_1 = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/DataClasses" + directory_name +"/original/"

    dir_src_2 = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/DataClasses/Classes01/photoshopped/"
    dir_dst_2 = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/DataClasses/" + directory_name + "/photoshopped/"
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


def target_process():
    img_name_photoshopped = []
    img_dir_photoshopped = BASE_PATH + 'DataClasses/Classes01' + os.path.sep + 'photoshopped'
    img_path_photoshopped = glob.glob(os.path.join(img_dir_photoshopped, "*.*"))
    for img in img_path_photoshopped:
        img = img.replace('/', '')
        img_name_photoshopped.append([img[71:],0])

    df_photoshopped_all = pd.DataFrame(img_name_photoshopped, columns=['img_name', 'target'])
    # shuffle the DataFrame rows
    df_photoshopped_all = df_photoshopped_all.sample(frac=1) 
    df_photoshopped = df_photoshopped_all.head(10000)

    img_name_original = []
    img_dir_original = BASE_PATH + 'DataClasses/Classes01' + os.path.sep + 'original'
    img_path_original = glob.glob(os.path.join(img_dir_original, "*.*"))
    for img in img_path_original:
        img = img.replace('/', '')
        img_name_original.append([img[67:], 1])

    df_original_all = pd.DataFrame(img_name_original, columns=['img_name', 'target'])
    
    # shuffle the DataFrame rows
    df_original_all = df_original_all.sample(frac=1)
    df_original = df_original_all.head(10000)
    
    df_list =[df_photoshopped,df_original]
    df = pd.concat(df_list)
    # shuffle the DataFrame rows
    df = df.sample(frac=1)  # df count :20000

    training_data, testing_data = train_test_split(df,test_size=0.20, random_state=33)

    training_data = df.sample(frac=0.8, random_state=25) # training count 16000
    #training_data['split']='train'

    testing_data = df.drop(training_data.index) # testing count 3643
    #testing_data['split']='test'

    df_data = [training_data, testing_data]
    df_model =pd.concat(df_data)

    df_model.to_excel('data_model.xlsx')

    return (training_data,testing_data)

    #eturn (training_data.count(),training_data.shape,training_data.head(10),
     #   testing_data.count(),testing_data.shape,testing_data.head(10))
    #return (df_photoshopped.count(), df.columns,df.head(10))
    #return (df_original_all.count(),df_original.count(),df_photoshopped_all.count(),
           # df_photoshopped.count(),df.count())




x_train, y_test= target_process()
IMG_DIR = r"/home/ubuntu/MLP/DATS6203-Final-Project/Data/DataClasses/Classes01/final/"

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(IMG_DIR)):
        if img in x_train['img_name'].tolist():
            path = os.path.join(IMG_DIR,img)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            label= x_train['target']
            training_data.append([np.array(img),np.array(label)])
    return training_data

def create_test_data():
    testing_data =[]
    for img in tqdm(os.listdir(IMG_DIR)):
        if img in y_test['img_name'].tolist():
            path = os.path.join(IMG_DIR,img)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            label= y_test['target']
            testing_data.append([np.array(img),np.array(label)])
    return testing_data

train =create_train_data()
test =create_test_data()

#train = train_data[:-500]
#test = train_data[-500:]

x_train =np.array([i[0] for i in train]).reshape(-1, IMG_SIZE,IMG_SIZE,1)
y_train =[i[1] for i in train]
x_test =np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test =[i[1] for i in test]



