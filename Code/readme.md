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

The outcome of Data preprocessing and modeling will different saved models with various perfromance. To evaluate each model, a dedicated Load_Predict_Measure.py file is created to 
load one of the save models from the modeling phase and perform the necessary steps to predict and measure its performance.

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


