'''
PartA. Data Engineering
'''

!pip install --upgrade --no-cache-dir gdown

from IPython.display import clear_output 
# clear_output()

!nvidia-smi

# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_3_06_TensorFlow2_Yolo_V1_V2_V3_BloodCell.git
# ! git pull origin master
! git pull origin main

# Toy MNIST Object Detection dataset download from Auther's Github repository
import gdown

google_path = 'https://drive.google.com/uc?id='
file_id = '1UjDF07l-fRyegqLRFO2LYxO5vGF12HAa'
output_name = '01_1K_MNIST.zip'
gdown.download(google_path+file_id,output_name,quiet=False)
# https://drive.google.com/file/d/1UjDF07l-fRyegqLRFO2LYxO5vGF12HAa/view?usp=sharing

import shutil
shutil.rmtree('/content/sample_data', ignore_errors=True)

!mkdir dataset

!unzip /content/01_1K_MNIST.zip -d /content/dataset
clear_output()

class_names = ['0', '1', '2',  '3', '4', 
          '5', '6', '7',  '8','9']

num_classes = len(class_names)

from os import path
from utils import tools

class Yolo_data(object):

    def __init__(self,
                 input_shape=(448, 448, 3),
                 class_names=[]):
        self.input_shape = input_shape
        self.grid_shape = input_shape[0]//64, input_shape[1]//64
        self.class_names = class_names
        self.class_num = len(class_names)
        self.file_names = None
        
    def read_file_to_dataset(
        self, img_path=None, label_path=None,
        label_format="labelimg",
        rescale=1/255,
        preprocessing=None,
        augmenter=None,
        aug_times=1,
        shuffle=True, seed=None,
        encoding="big5",
        thread_num=10):
        
        img_data, label_data, path_list = tools.read_file(
            img_path=img_path, 
            label_path=label_path,
            label_format=label_format,
            size=self.input_shape[:2], 
            grid_shape=self.grid_shape,
            class_names=self.class_names,
            rescale=rescale,
            preprocessing=preprocessing,
            augmenter=augmenter,
            aug_times=aug_times,
            shuffle=shuffle, seed=seed,
            encoding=encoding,
            thread_num=thread_num)
        self.file_names = path_list

        return img_data, label_data

    def vis_img(self, img, label_data,
                conf_threshold=0.5,
                show_conf=True,
                nms_mode=0,
                nms_threshold=0.5,
                nms_sigma=0.5,
                **kwargs):

        return tools.vis_img(
                             img, 
                             label_data, 
                             class_names=self.class_names,
                             conf_threshold=conf_threshold,
                             show_conf=show_conf,
                             nms_mode=nms_mode,  
                             nms_threshold=nms_threshold,
                             nms_sigma=nms_sigma,
                             **kwargs)

yolo_data = Yolo_data(class_names=class_names)

img_path   = "/content/dataset/mnist_train"
label_path = "/content/dataset/xml_train"

train_img, train_label = yolo_data.read_file_to_dataset(
    img_path, label_path,
    label_format="labelimg",
    thread_num=50,
    shuffle=False)

for i in range(5):
    yolo_data.vis_img(
        train_img[i], train_label[i],
        show_conf=False)

img_path   = "/content/dataset/mnist_val"
label_path = "/content/dataset/xml_val"

test_img, test_label = yolo_data.read_file_to_dataset(
    img_path, label_path,
    label_format="labelimg",
    thread_num=50,
    shuffle=False)

valid_img  = test_img
valid_label = test_label

for i in range(5):
    yolo_data.vis_img(
        valid_img[i], valid_label[i],
        show_conf=False)

'''
Part B. Model Engineering
'''

'''
1. Import Libraries
'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization

import tensorflow as tf
# from tensorflow.keras.layers import BatchNormalization as BN

'''
2. Hyperparameters
'''
n_epoch = 10

'''
3. DropBlock
'''

from tensorflow.keras import backend as K

keep_prob = 0.7

def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)))


class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True,name=None, **kwargs):
        super(DropBlock2D, self).__init__(name="DropBlock2D")
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.names = name
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale
        super(DropBlock2D, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update( {"block_size": self.block_size,"keep_prob": self.keep_prob,"name": self.names })

        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output *tf.cast(tf.size(mask), dtype=tf.float32)  / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.cast(self.w, dtype=tf.float32), tf.cast(self.h, dtype=tf.float32)
        
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask

'''
4. Convolutional
'''

def Conv2D_BN_Leaky(input_tensor, *args):
    output_tensor = Conv2D(*args, 
                           padding='same',
                           kernel_initializer='he_normal')(input_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = LeakyReLU(alpha=0.1)(output_tensor)
    return output_tensor

'''
5. Residual Block
Not Used
'''

'''
6. Backbone
'''
def Backbone_darknet(input_tensor):
    conv1 = Conv2D_BN_Leaky(input_tensor, 64, 7, 2)
    conv1 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D_BN_Leaky(pool1, 192, 3)
    conv2 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D_BN_Leaky(pool2, 128, 1)
    conv3 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv3)
    conv3 = Conv2D_BN_Leaky(conv3, 256, 3)
    conv3 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv3)
    conv3 = Conv2D_BN_Leaky(conv3, 256, 1)
    conv3 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv3)
    conv3 = Conv2D_BN_Leaky(conv3, 512, 3)
    conv3 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = pool3
    for _ in range(4):
        conv4 = Conv2D_BN_Leaky(conv4, 256, 1)
        conv4 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv4)
        conv4 = Conv2D_BN_Leaky(conv4, 512, 3)
        conv4 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv4)
    conv4 = Conv2D_BN_Leaky(conv4, 1024, 3)
    conv4 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D_BN_Leaky(pool4, 512, 1)
    conv5 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv5)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv5 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv5)
    conv5 = Conv2D_BN_Leaky(conv5, 512, 1)
    conv5 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv5)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv5 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv5)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv5 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv5)
    conv5 = Conv2D_BN_Leaky(conv5, 1024, 3, 2)
    conv5 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv5)
    
    conv6 = Conv2D_BN_Leaky(conv5, 1024, 3)
    conv6 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv6)
    conv6 = Conv2D_BN_Leaky(conv6, 1024, 3)
    conv6 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv6)
    
    return conv6

'''
7. Neck
'''
def yolo_neck(input_shape=(448, 448, 3),
              pretrained_darknet=None):
    inputs = Input(input_shape)
    darknet = Model(inputs, Backbone_darknet(inputs))
    
    if pretrained_darknet is not None:
        darknet.set_weights(pretrained_darknet.get_weights())
    
    return darknet

'''
8. Head
'''
def yolo_head(model_body, bbox_num=2, class_num=10):
    inputs = model_body.input
    output = model_body.output

    xywhc_output = Conv2D(5*bbox_num, 1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal')(output)
    p_output = Conv2D(class_num, 1,
                      padding = 'same',
                      activation='softmax',
                      kernel_initializer='he_normal')(output)

    outputs = concatenate([xywhc_output, p_output], axis=3)
    
    model = Model(inputs, outputs)

    return model

import tensorflow as tf
import numpy as np

epsilon = 1e-07

'''
9. Intersection over Union
'''
def cal_iou(xywh_true, xywh_pred, grid_shape):
    grid_shape = np.array(grid_shape[::-1])
    xy_true = xywh_true[..., 0:2]/grid_shape # N*S*S*1*2
    wh_true = xywh_true[..., 2:4] # N*S*S*1*2

    xy_pred = xywh_pred[..., 0:2]/grid_shape # N*S*S*B*2
    wh_pred = xywh_pred[..., 2:4]
    
    half_xy_true = wh_true / 2. # N*S*S*1*2
    mins_true    = xy_true - half_xy_true
    maxes_true   = xy_true + half_xy_true

    half_xy_pred = wh_pred / 2. # N*S*S*B*2
    mins_pred    = xy_pred - half_xy_pred
    maxes_pred   = xy_pred + half_xy_pred       
    
    intersect_mins  = tf.maximum(mins_pred,  mins_true) # N*S*S*B*2
    intersect_maxes = tf.minimum(maxes_pred, maxes_true)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1] # N*S*S*B
    
    true_areas = wh_true[..., 0] * wh_true[..., 1] # N*S*S*1
    pred_areas = wh_pred[..., 0] * wh_pred[..., 1] # N*S*S*B

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = (intersect_areas + epsilon)/(union_areas + epsilon)
    
    return iou_scores

'''
10. Yolo Loss Function
'''
def wrap_yolo_loss(grid_shape,
                   bbox_num,
                   class_num,
                   binary_weight=1,
                   loss_weight=[1, 1, 1, 1],
                   ):
    def yolo_loss(y_true, y_pred):
        xywhc_true = tf.reshape(
            y_true[..., :-class_num],
            (-1, *grid_shape, 1, 5)) # N*S*S*1*5
        xywhc_pred = tf.reshape(
            y_pred[..., :-class_num],
            (-1, *grid_shape, bbox_num, 5)) # N*S*S*B*5

        iou_scores = cal_iou(xywhc_true, xywhc_pred, grid_shape) # N*S*S*B
        
        response_mask = tf.one_hot(tf.argmax(iou_scores, axis=-1),
                                   depth=bbox_num,
                                   dtype=xywhc_true.dtype) # N*S*S*B
        response_mask_exp = tf.expand_dims(response_mask, axis=-1) # N*S*S*B*1

        has_obj_mask = xywhc_true[..., 4] # N*S*S*1
        has_obj_mask_exp = tf.expand_dims(has_obj_mask, axis=-1) # N*S*S*1*1
        
        no_obj_mask = 1 - has_obj_mask*response_mask # N*S*S*B

        xy_true = xywhc_true[..., 0:2] # N*S*S*1*2
        xy_pred = xywhc_pred[..., 0:2] # N*S*S*B*2

        wh_true = tf.maximum(xywhc_true[..., 2:4], epsilon) # N*S*S*1*2
        wh_pred = tf.maximum(xywhc_pred[..., 2:4], epsilon) # N*S*S*B*2

        c_pred = xywhc_pred[..., 4] # N*S*S*B
        
        xy_loss = tf.reduce_sum(
            tf.reduce_mean(
                has_obj_mask_exp # N*S*S*1*1
                *response_mask_exp # N*S*S*B*1
                *tf.square(xy_true - xy_pred), # N*S*S*B*2
                axis=0))

        wh_loss = tf.reduce_sum(
            tf.reduce_mean(
                has_obj_mask_exp # N*S*S*1*1
                *response_mask_exp # N*S*S*B*1
                *tf.square(tf.sqrt(wh_true) - tf.sqrt(wh_pred)), # N*S*S*B*2
                axis=0))

        has_obj_c_loss = tf.reduce_sum(
            tf.reduce_mean(
                has_obj_mask # N*S*S*1
                *response_mask # N*S*S*B
                *tf.square(iou_scores - c_pred), # N*S*S*B
                axis=0))

        no_obj_c_loss = tf.reduce_sum(
            tf.reduce_mean(
                no_obj_mask # N*S*S*1
                *(tf.square(0 - c_pred)), # N*S*S*B
                axis=0))
        
        c_loss = has_obj_c_loss + binary_weight*no_obj_c_loss

        p_true = y_true[..., -class_num:] # N*S*S*C
        p_pred = y_pred[..., -class_num:] # N*S*S*C
        p_pred = tf.clip_by_value(p_pred, epsilon, 1 - epsilon)
        p_loss = -tf.reduce_sum(
            tf.reduce_mean(
                has_obj_mask # N*S*S*1
                *p_true*tf.math.log(p_pred), # N*S*S*C
                axis=0))

        loss = (loss_weight[0]*xy_loss
                + loss_weight[1]*wh_loss
                + loss_weight[2]*c_loss
                + loss_weight[3]*p_loss)

        return loss

    return yolo_loss

'''
11. [Opt] Define Custom Metrics 1
Object Accuracy
'''
from tensorflow.keras.metrics import binary_accuracy

epsilon = 1e-07

def wrap_obj_acc(grid_shape, bbox_num, class_num):
    def obj_acc(y_true, y_pred):
        xywhc_true = tf.reshape(
            y_true[..., :-class_num],
            (-1, *grid_shape, 1, 5)) # N*S*S*1*5
        xywhc_pred = tf.reshape(
            y_pred[..., :-class_num],
            (-1, *grid_shape, bbox_num, 5)) # N*S*S*B*5
        
        c_true = xywhc_true[..., 4] # N*S*S*1
        c_pred = tf.reduce_max(xywhc_pred[..., 4], # N*S*S*B
                               axis=-1,
                               keepdims=True) # N*S*S*1

        bi_acc = binary_accuracy(c_true, c_pred)

        return bi_acc
    return obj_acc

'''
12. [Opt] Define Custom Metrics 2
Mean IOU
'''
def wrap_mean_iou(grid_shape, bbox_num, class_num):
    def mean_iou(y_true, y_pred):
        xywhc_true = tf.reshape(
            y_true[..., :-class_num],
            (-1, *grid_shape, 1, 5)) # N*S*S*1*5
        xywhc_pred = tf.reshape(
            y_pred[..., :-class_num],
            (-1, *grid_shape, bbox_num, 5)) # N*S*S*B*5

        has_obj_mask = xywhc_true[..., 4] # N*S*S*1
        
        iou_scores = cal_iou(xywhc_true, xywhc_pred, grid_shape) # N*S*S*B
        iou_scores = tf.reduce_max(iou_scores, axis=-1, keepdims=True) # N*S*S*1
        iou_scores = iou_scores*has_obj_mask # N*S*S*1

        num_p = tf.reduce_sum(has_obj_mask)

        return tf.reduce_sum(iou_scores)/(num_p + epsilon)
    return mean_iou

'''
13. [Opt] Define Custom Metrics 3
Class Accuracy
'''
def wrap_class_acc(grid_shape, bbox_num, class_num):
    def class_acc(y_true, y_pred):
        xywhc_true = tf.reshape(
            y_true[..., :-class_num],
            (-1, *grid_shape, 5)) # N*S*S*5

        has_obj_mask = xywhc_true[..., 4] # N*S*S

        pi_true = tf.argmax(y_true[..., -class_num:], # N*S*S*C
                            axis=-1) # N*S*S
        pi_pred = tf.argmax(y_pred[..., -class_num:], # N*S*S*C
                            axis=-1) # N*S*S

        equal_mask = tf.cast(pi_true == pi_pred,
                             dtype=y_true.dtype) # N*S*S
        equal_mask = equal_mask*has_obj_mask # N*S*S

        num_p = tf.reduce_sum(has_obj_mask)

        return tf.reduce_sum(equal_mask)/(num_p + epsilon)
    return class_acc

'''
14. Build Class for Model, Loss, Metrics
'''
class Yolo(object):

    def __init__(self,
                 input_shape=(448, 448, 3),
                 class_names=[]):
        self.input_shape = input_shape
        self.grid_shape = input_shape[0]//64, input_shape[1]//64
        self.bbox_num = 2
        self.class_names = class_names
        self.class_num = len(class_names)
        self.model = None
        self.file_names = None
        
    '''
    15. Model Create
    '''
    def create_model(self,
                     bbox_num=2,
                     pretrained_weights=None,
                     pretrained_backbone=None):
        
        model_body = yolo_neck(self.input_shape,
                               pretrained_backbone)

        self.model = yolo_head(model_body,
                               bbox_num,
                               self.class_num)
         
        if pretrained_weights is not None:
            self.model.load_weights(pretrained_weights)
        self.grid_shape = self.model.output.shape[1:3]
        self.bbox_num = bbox_num

    '''
    16. Loss Create
    '''
    def loss(self,
             binary_weight,
             loss_weight=[5, 5, 1, 1]):
        
        if isinstance(loss_weight, dict):
            loss_weight_list = []
            loss_weight_list.append(loss_weight["xy"])
            loss_weight_list.append(loss_weight["wh"])
            loss_weight_list.append(loss_weight["conf"])
            loss_weight_list.append(loss_weight["prob"])
            loss_weight = loss_weight_list
        
        return wrap_yolo_loss(
            grid_shape=self.grid_shape,
            bbox_num=self.bbox_num, 
            class_num=self.class_num,
            binary_weight=binary_weight,
            loss_weight=loss_weight,
            )
    
    '''
    17. Metrics Create
    '''
    def metrics(self, type="obj_acc"):
        
        metrics_list = []     
        if "obj" in type:
            metrics_list.append(
                wrap_obj_acc(
                    self.grid_shape, 
                    self.bbox_num, 
                    self.class_num))
        if "iou" in type:
            metrics_list.append(
                wrap_mean_iou(
                    self.grid_shape, 
                    self.bbox_num, 
                    self.class_num))
        if "class" in type:
            metrics_list.append(
                wrap_class_acc(
                    self.grid_shape, 
                    self.bbox_num, 
                    self.class_num))
        
        return metrics_list

yolo = Yolo(class_names=class_names)

'''
18. Get anchor boxes
Not Used
'''

'''
19. Build NN model
from class
'''
yolo.create_model()
yolo.model.summary()

'''
Callback function
20. Learning Rate Scheduling
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
21. Define Optimizer
'''
from tensorflow.keras.optimizers import SGD, Adam

optimizer = Adam(learning_rate=1e-4)

'''
22. Loss Function
from YOLO class
'''
from utils.tools import get_class_weight

binary_weight = get_class_weight(
    train_label[..., 4:5],
    method='binary'
    )
print(binary_weight)

loss_weight = {
    "xy":5,
    "wh":5,
    "conf":1,
    "prob":1
    }

loss_fn = yolo.loss(
    binary_weight=binary_weight,
    loss_weight=loss_weight
    )
'''
23. Build Metrics
from Yolo Class
'''
metrics = yolo.metrics("obj+iou+class")

'''
24. Model Compilation
'''
yolo.model.compile(
    optimizer = optimizer,
    #optimizer=SGD(learning_rate=1e-10, momentum=0.9, decay=5e-4),
    loss = loss_fn,
    metrics = metrics
    )
import time

start_time = time.time()

'''
25. Model Training and Validation
'''
train_history = yolo.model.fit(
    train_img,
    train_label,
    epochs = n_epoch,
    batch_size=5,
    verbose=1,
    validation_data=(valid_img, valid_label),
    callbacks=[callback]
    )

'''
26. Predict and evaluate
'''
prediction = yolo.model.predict(test_img)

for i in range(len(test_img)):
    yolo_data.vis_img(
        test_img[i],
        prediction[i],
        conf_threshold=0.5,
        nms_mode=2,
        )

finish_time = time.time()

print(int(finish_time - start_time), "Sec")

'''
27. Show score table
'''

from utils.measurement import create_score_mat

create_score_mat(
    test_label,
    prediction,
    class_names=yolo.class_names,
    nms_mode=1,
    nms_threshold=0.5,
    conf_threshold=0.5,
    version=1)

