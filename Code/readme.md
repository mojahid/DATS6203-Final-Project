## Code Structure 
The code files are as follows:

* Data_Preprocessing.py
* Modeling_CNN.py
* Modeling_Pretrained_CNN.py
* Load_Predict_Measure.py
* Hard_Voting_Committee.py
* MLP_Ensembling.py

The above files has the right order to perform the following function after the raw data is downloaded as per the instrcutions in the Data folder.
The code files are modular and will run independenly based on data generated from other files. Before digging into the code, please read the readme file under the Data folder first.

## Background
The project aims to build a Convolutional Nerual Network model(s) to detect photoshopped (or genetally) manipulated images. To do so it was identified that plain images
with no filter or feature extracted will not be suffcient to train the network. After a quick research, the following feature extraction/filters were decided to be analyzed:

* [Error Level Analysis](https://en.wikipedia.org/wiki/Error_level_analysis)
* [Noise Analysis](https://en.wikipedia.org/wiki/Image_noise)
* [Gradinet Analysis](https://en.wikipedia.org/wiki/Image_gradient)

Following the project pipeline, each image in the original dataset will be transformed to the three filters to create three more datasets each is holding a new image with the
filter applied, functions for preprocessing the data and create filters are all in the Data_Preprocessing.py.

Due to the experimental nature of the project, different CNN architectures were assessed including transfer learning from known pre-trained models (i.e. VGG19) and to facilitate
such assessment, two separate python files were created Modeling_CNN.py and Modeling_Pretrained_CNN.py were the Modeling_CNN.py is used to configure, compile and train CNN models 
based on one or more datasets created by the preprocessing steps and the Modeling_Pretrained_CNN.py is used to configure, compile and train pre-trained models on similar dataset.
Each model created by any of these files will be saved for furthet analysis and usage.

The outcome of Data preprocessing and modeling will different saved models with various perfromance. To evaluate each model, a dedicated Load_Predict_Measure.py file is created to load one of the save models from the modeling phase and perform the necessary steps to predict and measure its performance.

Finally to achieve the best performance, the project will elect the top models to perform model ensembling which is performed by the following two different methods:
* Hard Voting: implemented in Hard_Voting_Committee.py where final decision about the photo evaluation (original or photoshopped) will be decided based on majority model voting
* MLP ensembling: implemented in MLP_ensembling.py where an MLP model is trained against different model output vs target value


## Data Preprocessing
The Data_Preprocessing.py, contains some exploratory functions to naviagte through the dataset, functions to delete unwated extension, functions to move the images to a 
folder structure required by the tensorflow [image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory) and 
functions to apply ELA, Noise and Gradient filters and save the new dataset in similar structure. 

![image](https://user-images.githubusercontent.com/34656794/144784849-09cb667d-a988-417a-9fae-fcfda864d4f0.png)
![image](https://user-images.githubusercontent.com/34656794/144784856-02367514-fd5d-405b-b1b6-5ff30b4c3205.png)

Note: Folder Structure has to be created manually and then corresponding functions can be invoked.

The following should be followed to complete the dataset setup according to the project flow:
1. Dowload the data as per the instrcutions in the Data section
2. Create Classes01, Classes02, Classes03 and Classes04 folders (can be any name but this is what is followed in  the project)
3. Under each of the directories created in step 2, create two directories: "original" and "photoshopped" as per the above image
4. Use update_image_directories() to copy images from the downlaoded folders to Classes01 folders (this function also exclude any unreadable or extra small images)
5. Create test folder structure: Test01, Test02, Test03 and Test04 and under each directory create the same two directories: "original" and "photoshopped"
6. Pick ~500 image from each class under ./Classes01/original and ./Classes01/photoshopped and paste them under ./Test01/original and ./Test01/photoshopped resp.
7. Call process_filter_on_data() 3 times to populate the new training dataset with appropriate filter applied

    `process_filter_on_data('ela', 'Classes02')`
    
    `process_filter_on_data('noise', 'Classes03')`
    
    `process_filter_on_data('lum', 'Classes04')`
 This will create the new dataset with the filter applied.
 8. Update the dir_src_1 and dir_src_2 in the process_filter_on_data() and repeat the process for the test data

    `process_filter_on_data('ela', 'Test02')`
    
    `process_filter_on_data('noise', 'Test03')`
    
    `process_filter_on_data('lum', 'Test04')`
    
    this will create the filtered images under the differnt sub-directories to be used in testing



## CNN Modeling

Both Modeling_CNN.py and Modeling_Pretrained_CNN.py are based on Tensorflow classes and using image_dataset_from_directory to automatically create the dataset from directories 
created in the pre-processing step. At the end of the training, matplotlib is used to plot training history of loss and accuracy across the epochs.

Key Configurations:
1. Using Adam as an optimizer with small learning rate of 0.0001 (which is adapted by the algorithm) was the best perfromance of the model and known for its computational efficiencies with images
2. Convolutional Layers are using ReLu as activation fucntion which is a non-linear function that corresponds to the non-linar nature of the images
3. Model uses Sigmoid with the last dense layer with one output and binary crossentropy as loss function (also tried Softmax, dense layer with 2 output and catigorical crossentropy)
4. Batch size is set to 48 to minimize any memory allocation problems with large images and mintain a good rate of updating the weight based on avg batch gradient
5. Batch normalization is used to keep output close to mean of 0 and variance of 1 which prevents the saturations of the activation function
6. Maxpooling is used to downsample the size of the image/feature maps moving deeper within the network
7. In the Modeling_CNN, augmentation is used as a layer to utilize GPU speed
8. Image pixels are rescaled in a seperate layer in the model
9. Using early stopping is configured in the model to avoid overfitting
10. Pre-trained model used is VGG19
11. To increase the generalization likelihood and dropout layer to drop 20% of the dense layer neurons  
12. Models are saved for further processing, usage and analysis
13. Training history will be plotted to show accurcy and loss throughout the epochs

![myplot_best_CNN_ELA](https://user-images.githubusercontent.com/34656794/144795616-120ac089-6911-42f9-a487-e4cd1de5e02b.png)


## Evaluation

There is an dedicated file to load, predict and measure the model performace. The model performace metrics used are using SK-learn to show accuracy, precision and recall for individual classes as well as weighted average based on the calculated confusion matrix. Also sea born package is used to display the confusion matrix as a heatmap.

![myplot_CM_Model11](https://user-images.githubusercontent.com/34656794/144795883-b300cca2-7b5f-44f2-8f51-77cea3a22e92.png)

## Ensembling

Finally after electing the best performing models, Hard_Voting_Committee.py and MLP_ensembling.py can be used to stack the models and pick the best performing output.
Both files contains the following:

1. Functions to load test data and prepare dataframes that will be used to concat the different model output
2. Function to load and evaluate saved model and populate its predictions in the dataframes
3. Using hard voting or MLP to evaluate the best output out of the stacked models
4. Display metrics and measrues for the combined/stacked models
