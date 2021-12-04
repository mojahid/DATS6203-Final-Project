import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import cv2
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.applications.vgg16 import VGG16

data_dir = r"/home/ubuntu/DATS6203-Final-Project/Data/Test/"
THRESHOLD = 0.5

def predict_func(test_ds,df):
  final_model = tf.keras.models.load_model('model_{}.h5'.format('P1'))
  print(final_model.summary)
  res = final_model.predict(test_ds)
  df['results'] = res
  df['results1'] = res
  df.loc[df.results >= THRESHOLD, "results1"] = 1
  df.loc[df.results < THRESHOLD, "results1"] = 0
  df['Acc'] = 0
  df.loc[df.results1 == df.Photostopped, "Acc"] = 1


  #df['result1'] = res2
  df.to_excel('results_{}.xlsx'.format('P1'), index=False)

orig_dir = data_dir + 'original'
ps_dir = data_dir + 'photoshopped'


files1 = []
files2 = []

for file in sorted(os.listdir(orig_dir)):
  im = cv2.imread(orig_dir+"/"+file)
  files1.append((file, 0, im.shape))

for file in sorted(os.listdir(ps_dir)):
  files2.append((file, 1, cv2.imread(ps_dir+"/"+file).shape))

df1 = pd.DataFrame(files1, columns=['FileName', 'Photostopped', 'Shape'])
df2 = pd.DataFrame(files2, columns=['FileName', 'Photostopped', 'Shape'])

df = pd.concat([df1, df2])

test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  label_mode="binary",
  shuffle=False,
  image_size=(150, 150),
  batch_size=1)

predict_func(test_ds,df)