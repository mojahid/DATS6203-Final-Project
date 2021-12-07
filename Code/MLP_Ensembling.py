import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# This file contains all the functions to perform model stacking regression based on the different trained model outputs
# It is based on the models that was saved frm the previous steps

# This code used to perform one the following:
# 1- MODE = 'build' stack best models and build a training dataset that can be used to train a linear MLP
#    This mode ends with having a csv file with all model outcome along with image size and target
#    This mode uses the training dataset with freezed models to capture all their output

# 2- MODE = 'train' use the CSV file obtained from the build MODE to train a multi-layer perceptron
#    The outcome of this function is the best evaluation from the different models
#    This mode ends with saving the model for the MLP that can be used later in a two phase prediction

# 3- MODE = 'test' use the test dataset and evaluate all output from all the different trained models
#    obtained in the previous steps and create dataframe as input for the stacked MLP model obtained in
#    the train MODE. Then evaluate the MLP against the concatenated outputs and measure the full output
#    against original target

# MODE is train, test or build
MODE = 'test'

# If MODE is test then test dataset should be used
if MODE == 'test':
  data_dir_01 = r"/home/ubuntu/DATS6203-Final-Project/Data/Test01/"
  data_dir_02 = r"/home/ubuntu/DATS6203-Final-Project/Data/Test02/"
  orig_dir = data_dir_01 + 'original'
  ps_dir = data_dir_01 + 'photoshopped'

# If MODE is not test then the dataset should be train
elif MODE == 'build':
  data_dir_01 = r"/home/ubuntu/DATS6203-Final-Project/Data/Classes01/"
  data_dir_02 = r"/home/ubuntu/DATS6203-Final-Project/Data/Classes02/"
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
  # some models used softmax with two output parameters while other models used sigmoid with binary classifier
  # To accomodate both, predict function will always take the second prediction of the softmax
  # (representing photoshopped images) if results has a shape of 2
  # if not then it takes the results as is
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
  df['Model-11'] = predict_func(test_ds_02, '11')
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

def train_stacked_model(df_all):
    """ This function process the dataframe that contains all models predictions to train a MLP
    Keyword arguments:
    df_all -- dataframe with all model predictions
    """
    train_features = df_all.copy()
    train_labels = train_features.pop('Photoshopped')
    train_features = train_features.iloc[:, 1:]

    model = Sequential()
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    check_point = tf.keras.callbacks.ModelCheckpoint('model_{}.h5'.format('stacked_01'),
                                                     monitor='accuracy',
                                                     save_best_only=True)

    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(opt,loss='BinaryCrossentropy', metrics=['accuracy'])
    class_weight = {0: 1, 1: 0.95}
    model.fit(train_features, train_labels, epochs=250, batch_size=250, verbose=1,
              callbacks=[check_point],
              validation_split=0.2, class_weight = class_weight)
    return

def predict_stacked_model(df_all,model):
    """ This function is used in the test MODE to process the the aggregate output from all models and use it to
    evaluate the MLP model completed in the train mode
    Keyword arguments:
    df_all -- dataframe with all model predictions
    model -- MLP saved model to be used in evaluation
    """
    final_model = tf.keras.models.load_model('model_{}.h5'.format(model))
    test_features = df_all.copy()
    test_labels = test_features.pop('Photoshopped')

    res = final_model.predict(test_features)
    results = list(map(lambda x: 0 if x < THRESHOLD else 1, res))
    return results

def measure_model(y_true,y_pred):
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
    ax.set_title(' Confusion Matrix for combined model')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['Original', 'Photoshopped'])
    plt.show()
    return

# MAIN

# The normal sequence of the MODE should be:
# 1- build: to build the dataset that will train the stacked MLP model
# 2- train: Use the dataset from the Build mode to train the MLP
# 3- test: use the test dataset and the trained models and MLP to evaluate output

if MODE == 'test':
    # Load training or test data based on the directory (Classes or Test)
    # This data will be used against the individual models predictions
    # Since the best performance was on ELA or normal input we have only used two datasets here
    # One that holds normal images and the other holding ELA images
    test_ds_01, test_ds_02 = load_data(data_dir_01, data_dir_02)

    # Prepre the dataframe that will aggregate all model predictions
    df = prepare_dataframe(orig_dir, ps_dir)

    # Now we have the training/test data ready and the dataframe that will hold different model prediction
    # we can load our best models, predict their output on test_ds_01 and test_ds_02, lof the prediction in a dataframe
    df_all = load_and_predict_models(df)

    # keep only required columns
    feature_cols = ['size-mb', 'Model-P3', 'Model-P1', 'Model-04', 'Model-07', 'Model-09', 'Model-11', 'Photoshopped']
    df_all = df_all[feature_cols]

    # Save all model predictions in csv
    df_all.to_csv('all_models_out_test.csv')

    # Use stacked model to predict final output
    df_all['Final_Results'] = predict_stacked_model(df_all, 'stacked_01')

    y_true = np.array(df_all['Photoshopped'])
    y_pred = np.array(df_all['Final_Results'])

    # measure model performance
    measure_model(y_true,y_pred)
    df_all.to_csv('Final_out_test.csv')

elif MODE =='build':
    # Load training or test data based on the directory Classes (Train dataset)
    # This data will be used against the individual models predictions
    # Since the best performance was on ELA or normal input we have only used two datasets here
    # One that holds normal images and the other holding ELA images
    test_ds_01, test_ds_02 = load_data(data_dir_01, data_dir_02)

    # Prepare the dataframe that will aggregate all model predictions on the training data
    df = prepare_dataframe(orig_dir, ps_dir)

    # Using the training dataset, evaluate all models predictions
    df_all = load_and_predict_models(df)

    # keep only required columns
    feature_cols = ['size-mb', 'Model-P3', 'Model-P1', 'Model-04', 'Model-07', 'Model-09', 'Model-11', 'Photoshopped']
    df_all = df_all[feature_cols]

    # Save data to CSV, this data will be used later in the training mode to train the MLP
    df_all.to_csv('all_models_out.csv')

else:
    # Load training dataset that contains all models prediction for the training dataset
    df_all = pd.read_csv('all_models_out.csv')

    # Train the MLP and save it to be used later in testing the overall stacking performance
    train_stacked_model(df_all)

