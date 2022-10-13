!pip install --upgrade --no-cache-dir gdown

from IPython.display import clear_output 
# clear_output()

!nvidia-smi

# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_3_06_TensorFlow2_Yolo_V1_V2_V3_BloodCell.git
# ! git pull origin master
! git pull origin main

# Mini-Imagenet dataset download from Auther's Github repository
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

from os import path

from collections.abc import Iterable
from tensorflow.keras.utils import Sequence

from utils import tools
# from .models import yolo_body, yolo_head

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import get_file

from functools import wraps
from functools import reduce

from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2


def compose(*funcs):
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_initializer': 'he_normal'}
    if kwargs.get('strides') == (2, 2):
        darknet_conv_kwargs['padding'] = 'valid'
    else:
        darknet_conv_kwargs['padding'] = 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for _ in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters//2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3, 3))(x)
    return x, y

WEIGHTS_PATH_DN_BODY    = "https://github.com/samson6460/tf2_YOLO/releases/download/1.0/tf_keras_yolov3_body.h5"
WEIGHTS_PATH_DN53_TOP   = "https://github.com/samson6460/tf2_YOLO/releases/download/Weights/tf_keras_darknet53_448_include_top.h5"
WEIGHTS_PATH_DN53_NOTOP = "https://github.com/samson6460/tf2_YOLO/releases/download/Weights/tf_keras_darknet53_448_no_top.h5"

def yolo_body(input_shape=(416, 416, 3),
              pretrained_darknet=None,
              pretrained_weights=None):
    """Create YOLO_V3 model CNN body in Keras."""
    inputs = Input(input_shape)
    darknet = Model(inputs, darknet_body(inputs))
    
    if pretrained_darknet is not None:
        darknet.set_weights(pretrained_darknet.get_weights())
    
    x, y1 = make_last_layers(darknet.output, 512)

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256)

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128)
    model = Model(inputs, [y1, y2, y3])

    if pretrained_weights is not None:
        if pretrained_weights == "pascal_voc":
            pretrained_weights = get_file(
                "tf_keras_yolov3_body.h5",
                WEIGHTS_PATH_DN_BODY,
                cache_subdir="models")
        model.load_weights(pretrained_weights)

    return model


def yolo_head(model_body, class_num=10, 
              anchors=[[0.89663461, 0.78365384],
                       [0.37500000, 0.47596153],
                       [0.27884615, 0.21634615],
                       [0.14182692, 0.28605769],
                       [0.14903846, 0.10817307],
                       [0.07211538, 0.14663461],
                       [0.07932692, 0.05528846],
                       [0.03846153, 0.07211538],
                       [0.02403846, 0.03125000]]):
    anchors = np.array(anchors)
    inputs = model_body.input
    output = model_body.output
    tensor_num = len(output)

    if len(anchors)%tensor_num > 0:
        raise ValueError(("The total number of anchor boxs"
                          " should be a multiple of the number(%s)"
                          " of output tensors") % tensor_num)    
    abox_num = len(anchors)//tensor_num

    outputs_list = []
    for tensor_i, output_tensor in enumerate(output):
        output_list = []
        start_i = tensor_i*abox_num
        for box in anchors[start_i:start_i + abox_num]:
            xy_output = DarknetConv2D(2, 1,
                            activation='sigmoid')(output_tensor)
            wh_output = DarknetConv2D(2, 1,
                            activation='exponential')(output_tensor)
            wh_output = wh_output * box
            c_output = DarknetConv2D(1, 1,
                            activation='sigmoid')(output_tensor)
            p_output = DarknetConv2D(class_num, 1,
                            activation='sigmoid')(output_tensor)
            output_list += [xy_output,
                            wh_output,
                            c_output,
                            p_output]

        outputs = concatenate(output_list, axis=-1)
        outputs_list.append(outputs)

    model = Model(inputs, outputs_list)    

    return model

# from .losses import wrap_yolo_loss

import tensorflow as tf
import numpy as np

epsilon = 1e-07

def cal_iou(xywh_true, xywh_pred, grid_shape):
    grid_shape = np.array(grid_shape[::-1])
    xy_true = xywh_true[..., 0:2]/grid_shape # N*S*S*1*2
    wh_true = xywh_true[..., 2:4]

    xy_pred = xywh_pred[..., 0:2]/grid_shape # N*S*S*B*2
    wh_pred = xywh_pred[..., 2:4]
    
    half_xy_true = wh_true / 2.
    mins_true    = xy_true - half_xy_true
    maxes_true   = xy_true + half_xy_true

    half_xy_pred = wh_pred / 2.
    mins_pred    = xy_pred - half_xy_pred
    maxes_pred   = xy_pred + half_xy_pred       
    
    intersect_mins  = tf.maximum(mins_pred,  mins_true)
    intersect_maxes = tf.minimum(maxes_pred, maxes_true)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = wh_true[..., 0] * wh_true[..., 1]
    pred_areas = wh_pred[..., 0] * wh_pred[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = (intersect_areas + epsilon)/(union_areas + epsilon)
    
    return iou_scores


def wrap_yolo_loss(grid_shape,
                   bbox_num,
                   class_num,
                   anchors,
                   binary_weight=1,
                   loss_weight=[1, 1, 1, 1],
                   ignore_thresh=.6,
                   use_focal_loss=False,
                   focal_loss_gamma=2):
    def yolo_loss(y_true, y_pred):
        panchors = tf.reshape(anchors, (1, 1, 1, bbox_num, 2))

        y_true = tf.reshape(
            y_true,
            (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*(5+C)
        y_pred = tf.reshape(
            y_pred,
            (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*(5+C)

        xywh_true = y_true[..., :4] # N*S*S*1*4
        xywh_pred = y_pred[..., :4] # N*S*S*B*4

        iou_scores = cal_iou(xywh_true, xywh_pred, grid_shape) # N*S*S*B

        response_mask = tf.one_hot(tf.argmax(iou_scores, axis=-1),
                                   depth=bbox_num,
                                   dtype=xywh_true.dtype) # N*S*S*B

        has_obj_mask = y_true[..., 4]*response_mask # N*S*S*B
        has_obj_mask_exp = tf.expand_dims(has_obj_mask, axis=-1) # N*S*S*B*1

        no_obj_mask = tf.cast(
            iou_scores < ignore_thresh,
            iou_scores.dtype) # N*S*S*B
        no_obj_mask = (1 - has_obj_mask)*no_obj_mask # N*S*S*B

        xy_true = y_true[..., 0:2] # N*S*S*1*2
        xy_pred = y_pred[..., 0:2] # N*S*S*B*2

        wh_true = tf.maximum(y_true[..., 2:4]/panchors, epsilon) # N*S*S*1*2
        wh_pred = y_pred[..., 2:4]/panchors
        
        wh_true = tf.math.log(wh_true) # N*S*S*B*2
        wh_pred = tf.math.log(wh_pred) # N*S*S*B*2

        c_pred = y_pred[..., 4] # N*S*S*B

        box_loss_scale = 2 - y_true[..., 2:3]*y_true[..., 3:4] # N*S*S*1*1

        xy_loss = tf.reduce_sum(
            tf.reduce_mean(
                has_obj_mask_exp # N*S*S*B*1
                *box_loss_scale # N*S*S*1*1
                *tf.square(xy_true - xy_pred), # N*S*S*B*2
                axis=0))

        wh_loss = tf.reduce_sum(
            tf.reduce_mean(
                has_obj_mask_exp # N*S*S*B*1
                *box_loss_scale # N*S*S*1*1
                *tf.square(wh_true - wh_pred), # N*S*S*B*2
                axis=0))

        if use_focal_loss:
            c_pred = tf.clip_by_value(c_pred, epsilon, 1 - epsilon)

            has_obj_c_loss = -tf.reduce_sum(
                tf.reduce_mean(
                has_obj_mask # N*S*S*B
                *((1 - c_pred)**focal_loss_gamma)
                *tf.math.log(c_pred),
                axis=0))
            
            no_obj_c_loss = -tf.reduce_sum(
                tf.reduce_mean(
                no_obj_mask # N*S*S*B
                *((c_pred)**focal_loss_gamma)
                *tf.math.log(1 - c_pred),
                axis=0))
        else:
            has_obj_c_loss = tf.reduce_sum(
                tf.reduce_mean(
                has_obj_mask # N*S*S*1
                *(tf.square(1 - c_pred)), # N*S*S*B
                axis=0))

            no_obj_c_loss = tf.reduce_sum(
                tf.reduce_mean(
                no_obj_mask # N*S*S*1
                *(tf.square(0 - c_pred)), # N*S*S*B
                axis=0))
        
        c_loss = has_obj_c_loss + binary_weight*no_obj_c_loss

        p_true = y_true[..., -class_num:] # N*S*S*1*C
        p_pred = y_pred[..., -class_num:] # N*S*S*B*C
        p_pred = tf.clip_by_value(p_pred, epsilon, 1 - epsilon)
        p_loss = -tf.reduce_sum(
            tf.reduce_mean(
                has_obj_mask_exp # N*S*S*B*1
                *(p_true*tf.math.log(p_pred)
                + (1 - p_true)*tf.math.log(1 - p_pred)), # N*S*S*B*C
                axis=0))
        
        regularizer = tf.reduce_sum(
            tf.reduce_mean(wh_pred**2, axis=0))*0.01

        loss = (loss_weight[0]*xy_loss
                + loss_weight[1]*wh_loss
                + loss_weight[2]*c_loss
                + loss_weight[3]*p_loss
                + regularizer)

        return loss

    return yolo_loss

# from .metrics import wrap_obj_acc, wrap_mean_iou
# from .metrics import wrap_class_acc, wrap_recall

import tensorflow as tf
from tensorflow.keras.metrics import binary_accuracy

epsilon = 1e-07

def wrap_obj_acc(grid_shape, bbox_num, class_num):
    def obj_acc(y_true, y_pred):
        y_true = tf.reshape(
            y_true,
            (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
        y_pred = tf.reshape(
            y_pred,
            (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C
        
        c_true = y_true[..., 4] # N*S*S*1
        c_pred = tf.reduce_max(y_pred[..., 4], # N*S*S*B
                               axis=-1,
                               keepdims=True) # N*S*S*1

        bi_acc = binary_accuracy(c_true, c_pred)

        return bi_acc
    return obj_acc


def wrap_mean_iou(grid_shape, bbox_num, class_num):
    def mean_iou(y_true, y_pred):
        y_true = tf.reshape(
            y_true,
            (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
        y_pred = tf.reshape(
            y_pred,
            (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C

        has_obj_mask = y_true[..., 4] # N*S*S*1
        
        xywh_true = y_true[..., :4] # N*S*S*1*4
        xywh_pred = y_pred[..., :4] # N*S*S*B*4

        iou_scores = cal_iou(xywh_true, xywh_pred, grid_shape) # N*S*S*B
        iou_scores = tf.reduce_max(iou_scores, axis=-1, keepdims=True) # N*S*S*1
        iou_scores = iou_scores*has_obj_mask # N*S*S*1

        num_p = tf.reduce_sum(has_obj_mask)

        return tf.reduce_sum(iou_scores)/(num_p + epsilon)
    return mean_iou


def wrap_class_acc(grid_shape, bbox_num, class_num):
    def class_acc(y_true, y_pred):
        y_true = tf.reshape(
            y_true,
            (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
        y_pred = tf.reshape(
            y_pred,
            (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C

        has_obj_mask = y_true[..., 4] # N*S*S*1

        pi_true = tf.argmax(y_true[..., -class_num:], # N*S*S*1*C
                            axis=-1) # N*S*S*1
        pi_pred = tf.argmax(y_pred[..., -class_num:], # N*S*S*B*C
                            axis=-1) # N*S*S*B
        
        equal_mask = tf.cast(pi_true == pi_pred,
                             dtype=y_true.dtype) # N*S*S*B
        equal_mask = equal_mask*has_obj_mask # N*S*S*B

        num_p = tf.reduce_sum(has_obj_mask)*bbox_num

        return tf.reduce_sum(equal_mask)/(num_p + epsilon)
    return class_acc



class Yolo(object):

    def __init__(self,
                 input_shape=(416, 416, 3),
                 class_names=[]):
        self.input_shape = input_shape
        self.grid_shape = input_shape[0]//32, input_shape[1]//32
        self.abox_num = 3
        self.class_names = class_names
        self.class_num = len(class_names)
        self.fpn_layers = 3
        self.anchors = None
        self.model = None
        self.file_names = None
        
    def create_model(self,
                     anchors=[[0.89663461, 0.78365384],
                              [0.37500000, 0.47596153],
                              [0.27884615, 0.21634615],
                              [0.14182692, 0.28605769],
                              [0.14903846, 0.10817307],
                              [0.07211538, 0.14663461],
                              [0.07932692, 0.05528846],
                              [0.03846153, 0.07211538],
                              [0.02403846, 0.03125000]],
                     backbone="full_darknet",
                     pretrained_weights=None,
                     pretrained_darknet="pascal_voc"):
        
        if isinstance(pretrained_darknet, str):
            pre_body_weights = pretrained_darknet
            pretrained_darknet = None
        else:
            pre_body_weights = None
        
        model_body = yolo_body(self.input_shape,
            pretrained_weights=pre_body_weights)

        if pretrained_darknet is not None:
            model_body.set_weights(pretrained_darknet.get_weights())
        self.model = yolo_head(model_body,
                               self.class_num,
                               anchors)
         
        if pretrained_weights is not None:
            self.model.load_weights(pretrained_weights)
        self.anchors = anchors
        self.grid_shape = self.model.output[0].shape[1:3]
        self.fpn_layers = len(self.model.output)
        self.abox_num = len(self.anchors)//self.fpn_layers

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
        
        grid_amp = 2**(self.fpn_layers - 1)
        grid_shape = (self.grid_shape[0]*grid_amp,
                      self.grid_shape[1]*grid_amp)
        img_data, label_data, path_list = tools.read_file(
            img_path=img_path, 
            label_path=label_path,
            label_format=label_format,
            size=self.input_shape[:2], 
            grid_shape=grid_shape,
            class_names=self.class_names,
            rescale=rescale,
            preprocessing=preprocessing,
            augmenter=augmenter,
            aug_times=aug_times,
            shuffle=shuffle, seed=seed,
            encoding=encoding,
            thread_num=thread_num)
        self.file_names = path_list

        label_list = [label_data]
        for _ in range(self.fpn_layers - 1):
            label_data = tools.down2xlabel(label_data)
            label_list.insert(0, label_data)

        return img_data, label_list

    def vis_img(self, img, *label_datas,
                conf_threshold=0.5,
                show_conf=True,
                nms_mode=0,
                nms_threshold=0.5,
                nms_sigma=0.5,
                **kwargs):

        return tools.vis_img(
                             img, 
                             *label_datas, 
                             class_names=self.class_names,
                             conf_threshold=conf_threshold,
                             show_conf=show_conf,
                             nms_mode=nms_mode,  
                             nms_threshold=nms_threshold,
                             nms_sigma=nms_sigma,
                             version=3,
                             **kwargs)

    def loss(self,
             binary_weight=1,
             loss_weight=[1, 1, 5, 1],
             ignore_thresh=0.6,
             use_focal_loss=False,
             focal_loss_gamma=2):

        if (not isinstance(binary_weight, Iterable)
            or len(binary_weight) != self.fpn_layers):
            binary_weight = [binary_weight]*self.fpn_layers
        
        if isinstance(loss_weight, dict):
            loss_weight_list = []
            loss_weight_list.append(loss_weight["xy"])
            loss_weight_list.append(loss_weight["wh"])
            loss_weight_list.append(loss_weight["conf"])
            loss_weight_list.append(loss_weight["prob"])
            loss_weight = loss_weight_list
        
        loss_list = []
        for fpn_id in range(self.fpn_layers):
            grid_amp = 2**(fpn_id)
            grid_shape = (self.grid_shape[0]*grid_amp,
                          self.grid_shape[1]*grid_amp)
            anchors_id = self.abox_num*fpn_id
            loss_list.append(wrap_yolo_loss(
                grid_shape=grid_shape,
                bbox_num=self.abox_num, 
                class_num=self.class_num,
                anchors=self.anchors[
                    anchors_id:anchors_id + self.abox_num],
                binary_weight=binary_weight[fpn_id],
                loss_weight=loss_weight,
                ignore_thresh=ignore_thresh,
                use_focal_loss=use_focal_loss,
                focal_loss_gamma=focal_loss_gamma))
        return loss_list
    
    def metrics(self, type="obj_acc"):
        
        
        metrics_list = [[] for _ in range(self.fpn_layers)]
        for fpn_id in range(self.fpn_layers):
            grid_amp = 2**(fpn_id)
            grid_shape = (self.grid_shape[0]*grid_amp,
                            self.grid_shape[1]*grid_amp)
            
            if "obj" in type:
                metrics_list[fpn_id].append(
                    wrap_obj_acc(
                        grid_shape,
                        self.abox_num, 
                        self.class_num))
            if "iou" in type:
                metrics_list[fpn_id].append(
                    wrap_mean_iou(
                        grid_shape,
                        self.abox_num, 
                        self.class_num))
            if "class" in type:
                metrics_list[fpn_id].append(
                    wrap_class_acc(
                        grid_shape,
                        self.abox_num, 
                        self.class_num))
        
        return metrics_list

yolo = Yolo(class_names=class_names)

img_path   = "/content/dataset/mnist_train"
label_path = "/content/dataset/xml_train"

train_img, train_label = yolo.read_file_to_dataset(
    img_path, label_path,
    thread_num=50,
    shuffle=False)

for i in range(5):
    yolo.vis_img(
        train_img[i], train_label[2][i],
        show_conf=False)

img_path   = "/content/dataset/mnist_val"
label_path = "/content/dataset/xml_val"

test_img, test_label = yolo.read_file_to_dataset(
    img_path, label_path,
    thread_num=50,
    shuffle=False)

valid_img  = test_img
valid_label = test_label

for i in range(5):
    yolo.vis_img(
        valid_img[i], valid_label[2][i],
        show_conf=False)

# Get anchor boxes

from utils.kmeans import kmeans, iou_dist, euclidean_dist
import numpy as np

all_boxes = train_label[2][train_label[2][..., 4] == 1][..., 2:4]
anchors = kmeans(
    all_boxes,
    n_cluster=9,
    dist_func=iou_dist,
    stop_dist=0.000001)

anchors = np.sort(anchors, axis=0)[::-1]
display(anchors)

import matplotlib.pyplot as plt

plt.scatter(all_boxes[..., 0], all_boxes[..., 1])
plt.scatter(anchors[..., 0],
            anchors[..., 1],
            c="red")
plt.show()

# Create model
anchors=[[0.26923078, 0.26923078],
         [0.20192307, 0.20192307],
         [0.19016773, 0.13461539],
         [0.13461539, 0.10096154],
         [0.10120743, 0.0673077 ],
         [0.10096154, 0.06188303],
         [0.0673077 , 0.05288462],
         [0.05288462, 0.05113026],
         [0.03365385, 0.03365385]]

yolo.create_model(anchors=anchors)
yolo.model.summary()

from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch <= 20:
        return lr
    elif epoch <= 70:
        return 3e-5
    else:
        return 1e-5

callback = LearningRateScheduler(scheduler)

from utils.tools import get_class_weight

binary_weight_list = []

for i in range(len(train_label)):
    binary_weight_list.append(
        get_class_weight(
        train_label[i][..., 4:5],
        method='binary'
        )
    )
print(binary_weight_list)

binary_weight_list = [0.1]*3

from tensorflow.keras.optimizers import SGD, Adam

ignore_thresh = 0.7
use_focal_loss = True

loss_weight = {
    "xy":1,
    "wh":1,
    "conf":5,
    "prob":1
    }

loss = yolo.loss(
    binary_weight_list,
    loss_weight=loss_weight,
    ignore_thresh=ignore_thresh,
    use_focal_loss=use_focal_loss,
    )

metrics = yolo.metrics("obj+iou+class")

yolo.model.compile(
    optimizer=Adam(lr=5e-5),
    #optimizer=SGD(lr=1e-10, momentum=0.9, decay=5e-4),
    loss=loss,
    metrics=metrics
    )

train_history = yolo.model.fit(
    train_img,
    train_label,
    epochs=50,
    batch_size=5,
    verbose=1,
    validation_data=(valid_img, valid_label),
    callbacks=[callback]
    )

# Predict and evaluate
prediction = yolo.model.predict(test_img, batch_size=10)

for i in range(len(test_img)):
    yolo.vis_img(
        test_img[i],
        prediction[2][i],
        prediction[1][i],
        prediction[0][i],
        conf_threshold=0.5,
        nms_mode=2,
        )

# Show score table

from utils.measurement import create_score_mat

create_score_mat(
    test_label[2],
    prediction[2],
    prediction[1],
    prediction[0],
    class_names=class_names,
    conf_threshold=0.5,
    nms_mode=2,
    nms_threshold=0.5,
    version=3)



