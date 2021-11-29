import cv2
import numpy as np
import matplotlib.pyplot as plt

def flip_crop_image(image,op1):
    image = cv2.flip(image, op1)

    #op1 = np.random.randint(1, 5)
    #image = image[op1:op1,op1:op1]
    return image

def hue_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = image[:, :, 2]
    v = np.where(v > 255, v-10 , v-10)
    image[:, :, 2] = v
    return image

def rotate_image(image):
    op1 = np.random.randint(-2, 2)
    image_height, image_width = image.shape[0:2]
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, op1, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    result = result[int(0.1*image_height):int(0.9*image_height), int(0.1*image_width):int(0.9*image_width) ]
    #result = cv2.resize(result,(300, 300))
    return result


def remove_noise_image(image):
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image

def contrast_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:, :, 2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255) for pixel in row] for row in image[:, :, 2]]
    image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

def brightness_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - 50
    v[v > lim] = 255
    v[v <= lim] += 50
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def gaus_blur_image(img):
    fsize = 5
    return cv2.GaussianBlur(img, (fsize, fsize), 0)

def augment_image(image):
    op1 = np.random.randint(1, 6)
    if(op1 == 1):
        image = flip_crop_image(image)
        image = brightness_image(image)
        return image
    elif(op1 == 2):
        image = flip_crop_image(image)
        image = rotate_image(image)
        return image
    elif (op1 == 3):
        image = flip_crop_image(image)
        image = contrast_image(image)
        return image
    elif (op1 == 4):
        image = flip_crop_image(image)
        image = remove_noise_image(image)
        return image
    elif (op1 ==5):
        image = rotate_image(image)
        return image
    elif (op1 ==6):
        image = flip_crop_image(image)
        image = gaus_blur_image(image)
        image = rotate_image(image)
        return image

    return image

