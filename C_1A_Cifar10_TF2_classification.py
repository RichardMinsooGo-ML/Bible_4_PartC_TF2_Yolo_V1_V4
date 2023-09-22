'''
PartA. Data Engineering
'''

!pip install --upgrade --no-cache-dir gdown

from IPython.display import clear_output 
# clear_output()

# [PASS] Git clone Feature map

'''
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo/5_TF2_UCF101_video_classification.git
! git pull origin master
# ! git pull origin main
'''

# Cifar 10 dataset download from Auther's Github repository
import gdown

google_path = 'https://drive.google.com/uc?id='
file_id = '18I06ymkUqKwEon4Dsb8GqkJORwPZZxLd'
output_name = 'Cifar_10.zip'
gdown.download(google_path+file_id,output_name,quiet=False)
# https://drive.google.com/file/d/18I06ymkUqKwEon4Dsb8GqkJORwPZZxLd/view?usp=sharing

import shutil
shutil.rmtree('/content/sample_data', ignore_errors=True)

!unzip /content/Cifar_10.zip -d /content/data
clear_output()
! rm /content/Cifar_10.zip

import os

class_names = ['airplanes', 'cars', 'birds',  'cats', 'deer', 
          'dogs', 'frogs', 'horses',  'ships','truck']

num_classes = len(class_names)

batch_size = 64

# path joining version for other paths


Datasets = "cifar_10_224_pixels"

# img_size = 224       # 224
img_size = 48
dst_dir_train = '/content/data/train/'
dst_dir_test  = '/content/data/test/'
    

num_train = sum([len(files) for r, d, files in os.walk(dst_dir_train)])
num_test  = sum([len(files) for r, d, files in os.walk(dst_dir_test)])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_ds = train_datagen.flow_from_directory(dst_dir_train,
                                                 target_size = (img_size, img_size),
                                                 batch_size = batch_size,
                                                 class_mode = 'sparse')
                                                 # class_mode = 'categorical')

test_ds = test_datagen.flow_from_directory(dst_dir_test,
                                            target_size = (img_size, img_size),
                                            batch_size = batch_size,
                                            class_mode = 'sparse')
                                            # class_mode = 'categorical')

'''
Part B. Model Engineering
'''

'''
01. Import Libraries
'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization

import tensorflow as tf
# from tensorflow.keras.layers import BatchNormalization as BN

'''
02. Hyperparameters
'''
n_epoch = 10

'''
03. DropBlock
Not used
'''

'''
04. Leaky Convolutional
'''

def Conv2D_BN_Leaky(input_tensor, *args):
    output_tensor = Conv2D(*args, 
                           padding='same',
                           kernel_initializer='he_normal')(input_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = LeakyReLU(alpha=0.1)(output_tensor)
    return output_tensor

'''
05. [Not Used] Mish Activation
'''

'''
06. [Not Used] Mish Convolutional
'''

'''
07. [Not Used] Residual Block
'''

'''
08. Backbone
'''
def Backbone_darknet(input_tensor):
    conv1 = Conv2D_BN_Leaky(input_tensor, 64, 7, 2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D_BN_Leaky(pool1, 192, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D_BN_Leaky(pool2, 128, 1)
    conv3 = Conv2D_BN_Leaky(conv3, 256, 3)
    conv3 = Conv2D_BN_Leaky(conv3, 256, 1)
    conv3 = Conv2D_BN_Leaky(conv3, 512, 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = pool3
    for _ in range(4):
        conv4 = Conv2D_BN_Leaky(conv4, 256, 1)
        conv4 = Conv2D_BN_Leaky(conv4, 512, 3)
    conv4 = Conv2D_BN_Leaky(conv4, 1024, 3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D_BN_Leaky(pool4, 512, 1)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv5 = Conv2D_BN_Leaky(conv5, 512, 1)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3, 2)
    
    conv6 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv6 = Conv2D_BN_Leaky(conv6, 1024, 3)
    
    return conv6

'''
09. [Not Used] SPP Module
'''

'''
10. YOLO Neck
'''
def yolo_neck(input_shape=(448, 448, 3)):
    inputs = Input(input_shape)
    darknet = Model(inputs, Backbone_darknet(inputs))
    
    return darknet

'''
11. Head
'''
def yolo_head(model_body, class_num=10):
    inputs = model_body.input
    output = model_body.output

    output   = Flatten()(output)
    output   = Dense(4096)(output)
    outputs = Dense(10, activation='softmax')(output)
    
    model = Model(inputs, outputs)

    return model

input_shape=(img_size, img_size, 3)

'''
12. [Not Used] Intersection over Union
'''

'''
13. Loss Function
--> Use Built in loss
'''

'''
14. [Opt] Define Custom Metrics 1
Object Accuracy
Not Used --> Built in Metrics
'''

'''
15. [Opt] Define Custom Metrics 2
Mean IOU
Not Used --> Built in Metrics
'''

'''
16. [Opt] Define Custom Metrics 3
Class Accuracy
Not Used --> Built in Metrics
'''

'''
17. Build Class for Model, Loss, Metrics
'''
class Yolo(object):

    def __init__(self,
                 input_shape=(img_size, img_size, 3),
                 class_names=[]):
        self.input_shape = input_shape
        self.class_names = class_names
        self.class_num = len(class_names)
        self.model = None
        self.file_names = None
        
    '''
    18. Model Create
    '''
    def create_model(self):
        
        model_body = yolo_neck(self.input_shape)

        self.model = yolo_head(model_body,
                               self.class_num)

    '''
    19. [Not Used] Loss Create
    '''
    
    '''
    20. [Not Used] Metrics Create
    '''

yolo = Yolo(class_names=class_names)

'''
21. Get anchor boxes
Not Used
'''

'''
22. Build NN model from class
'''
yolo.create_model()
yolo.model.summary()

'''
23. Define Optimizer
'''
from tensorflow.keras.optimizers import SGD, Adam

optimizer = Adam(learning_rate=1e-4)

'''
Callback function
24. Learning Rate Scheduling
'''
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch <= 20:
        return lr
    elif epoch <= 70:
        return 3e-5
    else:
        return 1e-5

callback = LearningRateScheduler(scheduler)

'''
25. Loss Function from YOLO class
--> Built in Loss
'''

'''
26. Build Metrics
--> Built in Metrics
'''
metrics=['accuracy']

'''
27. Model Compilation
'''
yolo.model.compile(
    optimizer = optimizer,
    loss = 'sparse_categorical_crossentropy',
    metrics = metrics
    )
model_name = 'cifar10_Darknet'


"""
from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
checkpoint = ModelCheckpoint('model_best_weights.h5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
model.fit_generator(X_train, Y_train, validation_data=(X_val, Y_val), 
      callbacks = [early_stop, checkpoint])
"""

steps_per_epoch  = int(num_train/batch_size)
validation_steps = int(num_test/batch_size)

import time

start_time = time.time()

'''
28. Model Training and Validation
'''
train_history = yolo.model.fit_generator(
    train_ds,
    steps_per_epoch = steps_per_epoch,
    epochs = n_epoch,
    validation_data = test_ds,
    verbose=1,
    validation_steps = validation_steps,
    callbacks=[callback]
    )

'''
29. Predict and evaluate
'''
yolo.model.evaluate_generator(test_ds, validation_steps)

finish_time = time.time()

print(int(finish_time - start_time), "Sec")

