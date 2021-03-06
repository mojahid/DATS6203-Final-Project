
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
import Image_Augmentation


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
    """ This function copies images from the original download folder and do the following:
    1- validate that image is readable before copying
    2- Copy readable and reasonably sized images from orignals folder to 'original' folder
    3- Copy all readable and reasonably sized photoshopped images in the sub-folders under one folder "photoshopped"
    4- Due to very big size of the dataset and the imbalance between original classes (~10K images) and photoshopped
       classes (~90K images), only 10K of the photoshopped images will be copied randomally
    Keyword arguments:
    img_data -- List of images coming get_photoshopped_images()
    """
    # This function requires the folders to be set before running
    dir_src = r"/home/ubuntu/DATS6203-Final-Project/Data/photoshops/"
    dir_dst = r"/home/ubuntu/DATS6203-Final-Project/Data/Classes01/photoshopped/"
    original_count = 0
    fake_count = 0
    for file in glob.iglob('%s/**/*.*' % dir_src, recursive=True):
        op1 = np.random.randint(1, 7)
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

# This function will not be used as augmentation is used as a layer in the CNN model
def augmentation_resampling():
    dir_src = r"/home/ubuntu/MLP/FinalProject/Data/Classes01/original/"
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

# Delete files with PNG extenstion
def del_png_files(path):
    counter = 0
    for file in glob.iglob('%s/**/*.*' % path, recursive=True):
        if file.endswith(".png"):
            os.remove(file)
            counter = counter + 1
    return counter

def create_noise_analysis(img):
    """ This creates a new image that contains the noise of an image by smoothing the image via a Denoising filter
    and then subtract the denoised version from the original to keep only the noise
    Keyword arguments:
    img -- image to be analyzed for noise
    """
    #img2 = cv2.medianBlur(img, 3)
    img2 = cv2.fastNlMeansDenoisingColored(img, h=1)
    return img - img2

def create_ela(img):
    """ This creates a new image that contains the ELA of an image by saving it as jpg with different quality and
    then retrieve a weighted absolute differentce between the original and saved image
    Keyword arguments:
    img -- image to be analyzed for Error Level Analysis
    """
    cv2.imwrite("temp.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img2 = cv2.imread("temp.jpg")
    cv2.imwrite("temp.jpg", img2, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img2 = cv2.imread("temp.jpg")
    diff = 15 * cv2.absdiff(img, img2)
    return diff

def create_luminance_grad(img):
    """ This creates a new image that contains the Gradient of an image by convoluting a vertical and horizontal kernels
    on the original image (Sobel 3x3) and then calculate the absolute value of the gradient magnetiude
    Keyword arguments:
    img -- image to be analyzed for Gradient Analysis
    """
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    src = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # compute gradients along the x and y axis, respectively
    #grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    #grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # compute the gradient magnitude and orientation
    #magnitude = np.sqrt((grad_x ** 2) + (grad_y ** 2))
    #orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 180

    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    lum_grad =  cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    #lum_grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    temp = np.zeros_like(img)
    temp[:,:,0] = lum_grad
    temp[:,:,1] = lum_grad
    temp[:,:,2] = lum_grad
    return temp

def process_filter_on_data(filter,directory_name):
    """ This functions apply the filer specified in the input parameter on the images in the dir_scr_1 and dir_src_2
    directories and copy the result with the same structure under the directory_name parameter
    NOTE: dir_src_1 and dir_src_2 has to be changed between Classes0x and Test0x manually depending if a test or train
    dataset is used.
    Keyword arguments:
    filter -- ela, noise or lum
    directory_name -- name of the directory where the new filtered images will be copied to and it has to be created
    manually in the file system
    """
    # Classes vs Test depending on Train or Test dataset
    #dir_src_1 = r"/home/ubuntu/DATS6203-Final-Project/Data/Classes01/original/"
    dir_src_1 = r"/home/ubuntu/DATS6203-Final-Project/Data/Test01/original/"
    dir_dst_1 = r"/home/ubuntu/DATS6203-Final-Project/Data/" + directory_name +"/original/"

    #dir_src_2 = r"/home/ubuntu/DATS6203-Final-Project/Data/Classes01/photoshopped/"
    dir_src_2 = r"/home/ubuntu/DATS6203-Final-Project/Data/Test01/photoshopped/"
    dir_dst_2 = r"/home/ubuntu/DATS6203-Final-Project/Data/" + directory_name + "/photoshopped/"

    # Iterate all files, apply filter and write in the specified directory
    for file in glob.iglob('%s/**/*.*' % dir_src_1, recursive=True):
        filename = os.path.basename(file)
        filename = dir_dst_1 + filename
        if filter == 'ela':
            cv2.imwrite(filename,create_ela(cv2.imread(file)))
        elif filter == 'noise':
            cv2.imwrite(filename, create_noise_analysis(cv2.imread(file)))
        elif filter =='lum':
            cv2.imwrite(filename, create_luminance_grad(cv2.imread(file)))

    for file in glob.iglob('%s/**/*.*' % dir_src_2, recursive=True):
        filename = os.path.basename(file)
        filename = dir_dst_2 + filename
        if filter == 'ela':
            cv2.imwrite(filename,create_ela(cv2.imread(file)))
        elif filter == 'noise':
            cv2.imwrite(filename, create_noise_analysis(cv2.imread(file)))
        elif filter =='lum':
            cv2.imwrite(filename, create_luminance_grad(cv2.imread(file)))

#image_name = '141vnd'
#image_name = '1085it'
image_name = '100d24'

# Get the photoshop images for a specific original image
img_data = get_photoshopped_images(image_name)

# Display the orginal and photoshopped images
plot_image_set(img_data)

# run the below once for the first copy from downloaded folders to the new structure
#original_count, fake_count = update_image_directories()
#print(original_count)
#print(fake_count)

# Delete PNG images
#print(del_png_files('/home/ubuntu/DATS6203-Final-Project/Data/Classes01/photoshopped/'))


# run the below for each filter and on each dataset
process_filter_on_data('noise', 'Test03')

#org_jpegCounter = len(glob.glob1('/home/ubuntu/DATS6203-Final-Project/Data/Classes01/original/',"*.jpg"))
#print(org_jpegCounter)
#org_pngCounter = len(glob.glob1('/home/ubuntu/DATS6203-Final-Project/Data/Classes01/original/',"*.png"))
#print(org_pngCounter)
#ps_jpegCounter = len(glob.glob1('/home/ubuntu/DATS6203-Final-Project/Data/Classes01/photoshopped/',"*.jpg"))
#print(ps_jpegCounter)
#ps_pngCounter = len(glob.glob1('/home/ubuntu/DATS6203-Final-Project/Data/Classes01/photoshopped/',"*.png"))
#print(ps_pngCounter)


#img_data2 = []
#img1 = cv2.imread('/home/ubuntu/MLP/FinalProject/Data/photoshops/100qo2/c69hhbs_0.jpg')
#img_data2.append(img1)
#img_data2.append(create_noise_analysis(img1))
#img1 = cv2.imread('/home/ubuntu/MLP/FinalProject/Data/originals/100qo2.jpg')
#img_data2.append(img1)
#img_data2.append(create_noise_analysis(img1))
#img1 = cv2.imread('/home/ubuntu/MLP/FinalProject/Data/Capture2.JPG')
#img_data2.append(img1)
#img_data2.append(create_noise_analysis(img1))

#plot_image_set(img_data2)

# Some image retrieval for documentation
img_data3 = []
img1 = cv2.imread('/home/ubuntu/DATS6203-Final-Project/Data/photoshops/106r2b/c6axsh8_0.jpg')
img_org = cv2.imread('/home/ubuntu/DATS6203-Final-Project/Data/originals/106r2b.jpg')
img_data3.append(img_org)
img_data3.append(img1)
img_data3.append(create_ela(img_org))
img_data3.append(create_ela(img1))
#img_data3.append(img1)
img_data3.append(create_noise_analysis(img_org))
img_data3.append(create_noise_analysis(img1))
img_data3.append(create_luminance_grad(img_org))
img_data3.append(create_luminance_grad(img1))
#img_data3.append(img1)
#plot_image_set(img_data3)

