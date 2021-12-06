import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# This file contains the necessary functions to evaluate different models and use hard voting to evaluate the
# final prediction

data_dir_01 = r"/home/ubuntu/DATS6203-Final-Project/Data/Test01/"
data_dir_02 = r"/home/ubuntu/DATS6203-Final-Project/Data/Test02/"

orig_dir = data_dir_01 + 'original'
ps_dir = data_dir_01 + 'photoshopped'

THRESHOLD = 0.5


def predict_func(test_ds,model):
  """ This function loads model (based on file name sequence) evaluate model against the data set given in test_ds
  Keyword arguments:
  test_ds -- tensor dataset holding the images
  model -- file sequence for the model to be loaded
  """
  final_model = tf.keras.models.load_model('model_{}.h5'.format(model))
  #print(final_model.summary)
  res = final_model.predict(test_ds)
  if res.shape[1] == 2:
    results = res[:,1]
  else:
    results = res
  results = list(map(lambda x : 0 if x < THRESHOLD else 1, results))
  return results



  #df['result1'] = res2
  df.to_excel('results_{}.xlsx'.format('08'), index=False)
  return df

def prepare_dataframe(orig_dir,ps_dir):
  """ This function prepares a new dataframe to hold information about the dataset (train or test)
  The dataframe will contain image_name, size and shape
  Keyword arguments:
  orig_dir -- directory for original photos (which can be train or test)
  ps_dir -- directory for photoshopped photos (which can be train or test)
  """
  files1 = []
  files2 = []

  # This function will loop each folder separately (original and photoshopped) and will populate the dataframe
  # with the necessary information and then merge both dataframe created out of each folder
  for file in sorted(os.listdir(orig_dir)):
    im = cv2.imread(orig_dir+"/"+file)
    files1.append((file, 0, os.path.getsize(orig_dir+"/"+file)/1024/1024, im.shape))

  for file in sorted(os.listdir(ps_dir)):
    files2.append((file, 1, os.path.getsize(ps_dir+"/"+file)/1024/1024, cv2.imread(ps_dir+"/"+file).shape))

  df1 = pd.DataFrame(files1, columns=['FileName', 'Photoshopped','size-mb', 'Shape'])
  df2 = pd.DataFrame(files2, columns=['FileName', 'Photoshopped','size-mb', 'Shape'])

  df = pd.concat([df1, df2])
  return df

def load_data(data_dir_01,data_dir_02):
  """ This function utilizes the built-in function image_dataset_from_directory() to load
  the images. It will load different image set (one with standard images and the other with the ELA applied filter)
  Based on the model and the mode, the appropriate dataset will be used.
  Keyword arguments:
  data_dir_01 -- directory for normal images (no filters) where the sub-directories originals and photoshops are located
  data_dir_01 -- directory for ELA images where the sub-directories originals and photoshops are located
  """
  test_ds_01 = tf.keras.utils.image_dataset_from_directory(
    data_dir_01,
    label_mode="binary",
    shuffle=False,
    image_size=(150, 150),
    batch_size=10)

  test_ds_02 = tf.keras.utils.image_dataset_from_directory(
    data_dir_02,
    label_mode="binary",
    shuffle=False,
    image_size=(150, 150),
    batch_size=10)
  return test_ds_01, test_ds_02

def load_and_predict_models(df):
  """ This function is custom built to call the predict_func for each of the selected models
  The function returns the same input dataframe with individual model prediction added
  each model prediction is in a seperate column with the model name
  Keyword arguments:
  df -- input dataframe that comes pre-populated with image name, size and shape
  """
  df['Model-P3'] = predict_func(test_ds_01, 'P3')
  df['Model-04'] = predict_func(test_ds_01, '04')
  df['Model-07'] = predict_func(test_ds_01, '07')
  df['Model-P1'] = predict_func(test_ds_02, 'P1')
  df['Model-09'] = predict_func(test_ds_02, '09')
  #df['Model-11'] = predict_func(test_ds_02, '11')
  return df

def measure_model(y_true, y_pred):
    """ This function is used in the test MODE and it uses the predicted and original targets to measure
    model perfomace and draw a heatmap for the confusion matrix
    Keyword arguments:
    y_true -- original target list
    y_pred -- predicted target list from the two phased prediction
    """
    confusion = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    print('****************')
    # print(confusion)
    ax = sns.heatmap(confusion, annot=True, cmap='Blues', fmt='d')
    ax.set_title(' Confusion Matrix for hard voted models')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['Original', 'Photoshopped'])
    plt.show()
    return


## MAIN
# Load test data
# This data will be used against the individual models predictions
# Since the best performance was on ELA or normal input we have only used two datasets here
# One that holds normal images and the other holding ELA images
test_ds_01, test_ds_02 = load_data(data_dir_01,data_dir_02)


# Prepre the dataframe that will aggregate all model predictions
df = prepare_dataframe(orig_dir, ps_dir)

# Now we have the training/test data ready and the dataframe that will hold different model prediction
# we can load our best models, predict their output on test_ds_01 and test_ds_02, lof the prediction in a dataframe
df_all = load_and_predict_models(df)

# Create a new column to hold the sum of all models
df['SUM'] = df['Model-P3'] + df['Model-04'] + df['Model-07'] + df['Model-P1'] + df['Model-09']

# Create a new column that checks if the SUM of all models >= 3
# which means (3 or more models predicted the image as photoshopped)
df['Final'] = 0
df.loc[df.SUM >= 3, "Final"] = 1

# Save the results for analysis
df.to_excel('results_{}.xlsx'.format('ALL'), index=False)

# Export the two lists that will be used for the confusion matrix
y_true = np.array(df['Photoshopped'])
y_pred = np.array(df['Final'])

# Measure the model
measure_model(y_true, y_pred)

