# Spring 2019 Deep Learning and Artificial Intelligence Project<br>Andrew Hosna

## Dataset
I took 1440 photos of a digital clock, 1 photo for each minute of the day, using my iPhone camera. Each photo was originally 4032 pixels by 3024 pixels but was cropped, resized, and converted to grayscale. Each cropped photo was then split in two 128 pixel by 128 pixel images, one image containing the hour and the other containing the minute, for use in the multi-input, multi-output models.

***

## Data Visualization
Many images of the processed data are provided in the Python Notebook that goes along with this project. Below are some examples:

Hours | Minutes
-----|---------
![11](/sample_data/h_1148_am.png)|![48](/sample_data/m_1148_am.png)
![12](/sample_data/h_1233_am.png)|![33](/sample_data/m_1233_am.png)
![11](/sample_data/h_1100_am.png)|![00](/sample_data/m_1100_am.png)
![5](/sample_data/h_538_am.png)|![38](/sample_data/m_538_am.png)
![1](/sample_data/h_116_am.png)|![16](/sample_data/m_116_am.png)
![1](/sample_data/h_125_am.png)|![25](/sample_data/m_125_am.png)

***

## Training, Validation, and Test Data
The following code was used to split each input dataset, the hour images and the minute images, into training, validation, and test sets. Each dataset was first split at a 70-30 ratio between training data and test data. The test data was then split in half to obtain a validation data set.

    from sklearn.model_selection import train_test_split

    train_hour_images, test_hour_images, train_hour_labels, test_hour_labels = train_test_split(dataset_hour_images, dataset_hour_labels, test_size=0.3, random_state=24)
    test_hour_images, val_hour_images, test_hour_labels, val_hour_labels = train_test_split(test_hour_images, test_hour_labels, test_size=0.5, random_state=24)

    train_minute_images, test_minute_images, train_minute_labels, test_minute_labels = train_test_split(dataset_minute_images, dataset_minute_labels, test_size=0.3, random_state=24)
    test_minute_images, val_minute_images, test_minute_labels, val_minute_labels = train_test_split(test_minute_images, test_minute_labels, test_size=0.5, random_state=24)

A random state of 24 was used to define the seed for the random number generator so that this split would be reproducible.

***

## Models

### Model 1
#### Single Conv2D Layer

    Base Model 1
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 128, 128, 1)       0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 126, 126, 64)      640       
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 1016064)           0         
    =================================================================
    Total params: 640
    Trainable params: 640
    Non-trainable params: 0
    _________________________________________________________________
    
  ![Base Model 1](/models/base_1.png)

    Model 1
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            (None, 128, 128, 1)  0                                            
    __________________________________________________________________________________________________
    input_3 (InputLayer)            (None, 128, 128, 1)  0                                            
    __________________________________________________________________________________________________
    model_1 (Model)                 (None, 1016064)      640         input_2[0][0]                    
                                                                 input_3[0][0]                    
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 13)           13208845    model_1[1][0]                    
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 60)           60963900    model_1[2][0]                    
    ==================================================================================================
    Total params: 74,173,385
    Trainable params: 74,173,385
    Non-trainable params: 0
    __________________________________________________________________________________________________
    
  ![Model 1](/models/model_1.png)

### Model 2
#### Two Conv2D Layers Followed by a MaxPooling2D Layer

    Base Model 2
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_4 (InputLayer)         (None, 128, 128, 1)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 126, 126, 64)      640       
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 124, 124, 64)      36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 62, 62, 64)        0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 246016)            0         
    =================================================================
    Total params: 37,568
    Trainable params: 37,568
    Non-trainable params: 0
    _________________________________________________________________
    
  ![Base Model 2](/models/base_2.png)

    Model 2
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_5 (InputLayer)            (None, 128, 128, 1)  0                                            
    __________________________________________________________________________________________________
    input_6 (InputLayer)            (None, 128, 128, 1)  0                                            
    __________________________________________________________________________________________________
    model_3 (Model)                 (None, 246016)       37568       input_5[0][0]                    
                                                                     input_6[0][0]                    
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 13)           3198221     model_3[1][0]                    
    __________________________________________________________________________________________________
    dense_4 (Dense)                 (None, 60)           14761020    model_3[2][0]                    
    ==================================================================================================
    Total params: 17,996,809
    Trainable params: 17,996,809
    Non-trainable params: 0
    __________________________________________________________________________________________________
    
  ![Model 2](/models/model_2.png)
    
### Model 3
#### Groups of Conv2D Layers Each Followed by a MaxPooling2D Layer

    Base Model 3
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_7 (InputLayer)         (None, 128, 128, 1)       0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 126, 126, 128)     1280      
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 124, 124, 128)     147584    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 62, 62, 128)       0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 60, 60, 64)        73792     
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 58, 58, 64)        36928     
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 56, 56, 64)        36928     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 28, 28, 64)        0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 26, 26, 32)        18464     
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 24, 24, 32)        9248      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 12, 12, 32)        0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 10, 10, 16)        4624      
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 8, 8, 16)          2320      
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 4, 4, 16)          0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 256)               0         
    =================================================================
    Total params: 331,168
    Trainable params: 331,168
    Non-trainable params: 0
    _________________________________________________________________
    
  ![Base Model 3](/models/base_3.png)
    
    Model 3
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_8 (InputLayer)            (None, 128, 128, 1)  0                                            
    __________________________________________________________________________________________________
    input_9 (InputLayer)            (None, 128, 128, 1)  0                                            
    __________________________________________________________________________________________________
    model_5 (Model)                 (None, 256)          331168      input_8[0][0]                    
                                                                     input_9[0][0]                    
    __________________________________________________________________________________________________
    dense_5 (Dense)                 (None, 13)           3341        model_5[1][0]                    
    __________________________________________________________________________________________________
    dense_6 (Dense)                 (None, 60)           15420       model_5[2][0]                    
    ==================================================================================================
    Total params: 349,929
    Trainable params: 349,929
    Non-trainable params: 0
    __________________________________________________________________________________________________
    
  ![Model 3](/models/model_3.png)

### Model 4
#### Simple Residual Network

    Base Model 4
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_10 (InputLayer)           (None, 128, 128, 1)  0                                            
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 128, 128, 1)  4           input_10[0][0]                   
    __________________________________________________________________________________________________
    re_lu_1 (ReLU)                  (None, 128, 128, 1)  0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 128, 128, 32) 320         re_lu_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 128, 128, 32) 128         conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    re_lu_2 (ReLU)                  (None, 128, 128, 32) 0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 128, 128, 1)  289         re_lu_2[0][0]                    
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, 128, 128, 1)  0           input_10[0][0]                   
                                                                     conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 128, 128, 1)  4           add_1[0][0]                      
    __________________________________________________________________________________________________
    re_lu_3 (ReLU)                  (None, 128, 128, 1)  0           batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 128, 128, 64) 640         re_lu_3[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 128, 128, 64) 256         conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    re_lu_4 (ReLU)                  (None, 128, 128, 64) 0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 128, 128, 1)  577         re_lu_4[0][0]                    
    __________________________________________________________________________________________________
    add_2 (Add)                     (None, 128, 128, 1)  0           add_1[0][0]                      
                                                                     conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, 128, 128, 1)  4           add_2[0][0]                      
    __________________________________________________________________________________________________
    re_lu_5 (ReLU)                  (None, 128, 128, 1)  0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 128, 128, 128 1280        re_lu_5[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, 128, 128, 128 512         conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    re_lu_6 (ReLU)                  (None, 128, 128, 128 0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 128, 128, 1)  1153        re_lu_6[0][0]                    
    __________________________________________________________________________________________________
    add_3 (Add)                     (None, 128, 128, 1)  0           add_2[0][0]                      
                                                                     conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, 128, 128, 1)  4           add_3[0][0]                      
    __________________________________________________________________________________________________
    re_lu_7 (ReLU)                  (None, 128, 128, 1)  0           batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 128, 128, 1)  0           re_lu_7[0][0]                    
    __________________________________________________________________________________________________
    max_pooling2d_6 (MaxPooling2D)  (None, 64, 64, 1)    0           dropout_1[0][0]                  
    __________________________________________________________________________________________________
    flatten_4 (Flatten)             (None, 4096)         0           max_pooling2d_6[0][0]            
    ==================================================================================================
    Total params: 5,171
    Trainable params: 4,715
    Non-trainable params: 456
    __________________________________________________________________________________________________
    
  ![Base Model 4](/models/base_4.png)
    
    Model 4
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_11 (InputLayer)           (None, 128, 128, 1)  0                                            
    __________________________________________________________________________________________________
    input_12 (InputLayer)           (None, 128, 128, 1)  0                                            
    __________________________________________________________________________________________________
    model_7 (Model)                 (None, 4096)         5171        input_11[0][0]                   
                                                                     input_12[0][0]                   
    __________________________________________________________________________________________________
    dense_7 (Dense)                 (None, 13)           53261       model_7[1][0]                    
    __________________________________________________________________________________________________
    dense_8 (Dense)                 (None, 60)           245820      model_7[2][0]                    
    ==================================================================================================
    Total params: 304,252
    Trainable params: 303,796
    Non-trainable params: 456
    __________________________________________________________________________________________________
    
  ![Model 4](/models/model_4.png)

***

## Evaluation

### Model 1
#### Single Conv2D Layer

    216/216 [==============================] - 1s 2ms/step
    Test Data Set
    Loss: 1.8534541571581806
    Hour Loss: 0.08%
    Minute Loss: 1.78%
    Hour Accuracy: 98.15%
    Minute Accuracy: 59.26%

    Predictions off by 5.4 minutes on average

### Model 2
#### Two Conv2D Layers Followed by a MaxPooling2D Layer

    216/216 [==============================] - 1s 6ms/step
    Test Data Set
    Loss: 2.0289083321889243
    Hour Loss: 0.08%
    Minute Loss: 1.95%
    Hour Accuracy: 98.15%
    Minute Accuracy: 65.74%

    Predictions off by 5.6 minutes on average

### Model 3
#### Groups of Conv2D Layers Each Followed by a MaxPooling2D Layer

    216/216 [==============================] - 4s 20ms/step
    Test Data Set
    Loss: 0.8320705537442807
    Hour Loss: 0.05%
    Minute Loss: 0.78%
    Hour Accuracy: 99.07%
    Minute Accuracy: 88.89%

    Predictions off by 4.3 minutes on average

### Model 4
#### Simple Residual Network

    216/216 [==============================] - 3s 16ms/step
    Test Data Set
    Loss: 1.791107698723122
    Hour Loss: 0.08%
    Minute Loss: 1.72%
    Hour Accuracy: 97.69%
    Minute Accuracy: 61.11%

    Predictions off by 6.7 minutes on average

***

## Benchmark



***

## Training Time

### Model 1
#### Single Conv2D Layer

    Total params: 74,173,385
    Trainable params: 74,173,385
    Non-trainable params: 0

    Train on 1008 samples, validate on 216 samples
    Epoch 1: 13 s (13 ms/step)
    Epochs 2-31 : 10 s (10 ms/step)
    Epoch 32: early stopping

    Total training time: 313 seconds (5 minutes 13 seconds)

### Model 2
#### Two Conv2D Layers Followed by a MaxPooling2D Layer

    Total params: 17,996,809
    Trainable params: 17,996,809
    Non-trainable params: 0

    Train on 1008 samples, validate on 216 samples
    Epoch 1: 9 s (9 ms/step)
    Epochs 2-37: 7 s (7 ms/step)
    Epoch 38: early stopping

    Total training time: 261 seconds (4 minutes 21 seconds)

### Model 3
#### Groups of Conv2D Layers Each Followed by a MaxPooling2D Layer

    Total params: 349,929
    Trainable params: 349,929
    Non-trainable params: 0

    Train on 1008 samples, validate on 216 samples
    Epoch 1: 20 s (20 ms/step)
    Epochs 2-68: 17 s (17 ms/step)
    Epoch 69: early stopping

    Total training time: 1159 seconds (19 minutes 19 seconds)

### Model 4
#### Simple Residual Network

    Total params: 304,252
    Trainable params: 303,796
    Non-trainable params: 456

    Train on 1008 samples, validate on 216 samples
    Epoch 1: 19 s (19 ms/step)
    Epochs 2-40: 14 s (14 ms/step)
    Epoch 41: early stopping

    Total training time: 565 seconds (5 minutes 25 seconds)

***

## Learning Curves

### Model 1
#### Single Conv2D Layer

Loss | Accuracy
-----|---------
![Model 1 Loss](/models/model_1_loss.png)|![Model 1 Accuracy](/models/model_1_acc.png)

### Model 2
#### Two Conv2D Layers Followed by a MaxPooling2D Layer

Loss | Accuracy
-----|---------
![Model 2 Loss](/models/model_2_loss.png)|![Model 2 Accuracy](/models/model_2_acc.png)

### Model 3
#### Groups of Conv2D Layers Each Followed by a MaxPooling2D Layer

Loss | Accuracy
-----|---------
![Model 3 Loss](/models/model_3_loss.png)|![Model 3 Accuracy](/models/model_3_acc.png)

### Model 4
#### Simple Residual Network

Loss | Accuracy
-----|---------
![Model 4 Loss](/models/model_4_loss.png)|![Model 4 Accuracy](/models/model_4_acc.png)

***

## Documentation
This project is documented within this file and the Python Notebook where this project was developed. The Python Notebook for this project provides guidance on how to run the project and reproduce the results. A link to it is provided below.

https://colab.research.google.com/drive/1Fb2PHnqgJ2Z_kK-3u9HQQ_Lt1SZufHan
