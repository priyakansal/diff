

import cv2
import numpy as np

import os 
from util.preprocess import preprocess_unityeyes_image
import json
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import albumentations as albu
from keras.layers import Input,add,Add, Dense,Conv2D,concatenate, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, Lambda, GlobalAveragePooling2D,Conv1D,MaxPooling1D
from keras.layers import BatchNormalization
from keras.models import Model, load_model, save_model
from keras.optimizers import Adam
import pandas as pd

def BatchActivate(x):
    x = BatchNormalization(axis=3, momentum=0.95,epsilon=0.0001)(x)
    x = Activation('elu')(x)
#    x = LeakyReLU(alpha=0.1)(x)

    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True,DILATION_VALUE=0):
    if DILATION_VALUE!=0:
        x = Conv2D(filters, size, strides=strides, padding=padding,dilation_rate=DILATION_VALUE)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False,DILATION_VALUE=0):
    
    x_side = convolution_block(blockInput, num_filters,(3,3),DILATION_VALUE=DILATION_VALUE)

    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) ,activation=True,DILATION_VALUE=DILATION_VALUE)
#    x = PReLU(shared_axes=[1, 2])(x)

    x= convolution_block(x, num_filters, (3,3), activation=True,DILATION_VALUE=DILATION_VALUE)

#    x = convolution_block(x, num_filters, (3,3), activation=True)

#    x = BatchActivate(x)
#    x=Squeeze_excitation_layer(x)
    x = Add()([x,x_side])
    if batch_activate:
        x = BatchActivate(x)
    return x


def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

                  padding='same', data_format0='channels_last'):
    if data_format0 == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]
        
    data_format ='channels_last' 

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):

        for i in range(2):
            if i == 0:

                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer
def lightweight_model(input_shape=(160,160,3),learning_rate=0.0001):
    
    inputs0 = Input(shape=input_shape)
    inputs1 = Input(shape=input_shape)

    cont0 = concatenate([inputs0,inputs1])
    conv0 = conv_block_simple(cont0, 16,prefix='64')
#    conv0 = residual_block(conv0,16, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],padding='same', data_format0='none')
    conv0 = residual_block(conv0,16, True)
    conv0 = residual_block(conv0,16, True)

    max0 =MaxPooling2D(3,padding='same',strides=2)(conv0)
    
    
    conv1 = conv_block_simple(max0, 16,prefix='128')
#    conv1 = residual_block(conv1,16, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],padding='same', data_format0='none')
    conv1 = residual_block(conv1,16, True)
    conv1 = residual_block(conv1,16, True)
    
    max1 =MaxPooling2D(2,padding='same')(conv1)
    
    conv2 = conv_block_simple(max1, 32,prefix='256')
#    conv2 = residual_block(conv2,32, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],padding='same', data_format0='none')
    conv2 = residual_block(conv2,32, True)
    conv2 = residual_block(conv2,32, True)
    max2 =MaxPooling2D(2,padding='same')(conv2)
    
    conv3 = conv_block_simple(max2, 64,prefix='512_0')
#    conv3 = rec_res_block(conv2,64, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],padding='same', data_format0='none')
    conv3 = residual_block(conv3,64, True)
    conv3 = residual_block(conv3,64, True)
    
    max3 =MaxPooling2D(2,padding='same')(conv3)
    
    conv4 = conv_block_simple(max3, 128,prefix='512_1')
#    conv4 = rec_res_block(conv4,128, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],padding='same', data_format0='none')
    conv4 = residual_block(conv4,128, True)
    conv4 = residual_block(conv4,128, True)
    max4 =MaxPooling2D(2,padding='same')(conv4)
    
    GA0 = GlobalAveragePooling2D()(max4)
    FC0 = Dense(128,activation='relu')(GA0)
    FC0 = Dropout(0.2)(FC0)
    FC0 = Dense(2)(FC0)
#    output= Conv2D(512, (1, 1), padding="same", kernel_initializer="he_normal")(max4)
    output_a=Activation('tanh')(FC0)
    
    model=Model([inputs0,inputs1],output_a)
    
    
    c = Adam(lr =learning_rate)

    model.compile(optimizer=c, loss=["mse"],
              metrics=['accuracy','mse'])

    
    return model

#fileName = np.array([ str(i)+'.jpg' for i in range(1,100938)])

train= pd.read_csv("/home/ubuntu/sabari/gazeEstimation/newDataset/train.csv")
valid= pd.read_csv("/home/ubuntu/sabari/gazeEstimation/newDataset/validation.csv")



datasetPath='/home/ubuntu/sabari/gazeEstimation/newDataset/dataset/train/'



IMG_HEIGHT = 48
IMG_WIDTH = 80



def __random_transform(img, masks):
        composition = albu.Compose([
#            albu.HorizontalFlip(),
            albu.Blur(blur_limit=7, p=0.5) ,
#            albu.VerticalFlip(),
            albu.GaussNoise(var_limit=(1.0, 5.0)),
            albu.RandomBrightnessContrast(p=0.2)
#            albu.Transpose(),
#            albu.RandomRotate90()
        ])
        
        composed = composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        
        return aug_img, aug_masks

def __augment_batch( img_batch, masks_batch):
    for i in range(img_batch.shape[0]):
        img_batch[i, ], masks_batch[i, ] = __random_transform(
            img_batch[i, ], masks_batch[i, ])
    
    return img_batch, masks_batch


def dataLoader(datasetPath,imageName,batchCount):
    
    image = cv2.imread(datasetPath+imageName)


    jsonPath =datasetPath+imageName[0:-3]+'json'


    with open(jsonPath) as f:
        json_data = json.load(f)
    
    
    eye_sample = preprocess_unityeyes_image(image, json_data)

    heatmaps = eye_sample['heatmaps']
    outputMap = heatmaps[16:33]
    
    preprocessedImage = eye_sample['img']
    
    gazeOP = eye_sample['gaze']
#    
#    if batchCount%2==1:
#        
#        preprocessedImage = cv2.flip(preprocessedImage,1)
#        outputMap = cv2.flip(outputMap,1)
#        gazeOP[1] = -gazeOP[1]
    
    return preprocessedImage,outputMap,gazeOP    
    

def generate_data(train_set, batch_size,shuffle=False):
    """Replaces Keras' native ImageDataGenerator."""
    i = 0
    
    
    train_ID=train_set
    
    imageNameList = train_ID.imageName.values
    gazePose0 = train_ID.gazePose0.values
    gazePose1 = train_ID.gazePose1.values
    folderName =train_ID.folderName.values
    
    
    

    batch_index=0
    while True:
        
        image_batch = np.zeros((batch_size,IMG_HEIGHT,IMG_WIDTH, 3))
        image_batch1 = np.zeros((batch_size,IMG_HEIGHT,IMG_WIDTH, 3))

#        Y_batch0 = np.zeros((batch_size,24,40,17))
        Y_gaze = np.zeros((batch_size,2))


        for b in range(batch_size):
            if i == len(train_ID) :
                i = 0
                #shuffle if u want to
                if shuffle:
                    train_set = train_set.sample(frac=1).reset_index(drop=True)

                    imageName = train_ID.imageName.values
                    gazePose0 = train_ID.gazePose0.values # yaw
                    gazePose1 = train_ID.gazePose1.values # pitch


            imageName = str(imageNameList[i])
            
            if len(imageName) ==1:
                imageNameNew ="00000"+imageName
            elif len(imageName)==2:
                imageNameNew ="0000"+imageName
            elif len(imageName)==3:
                imageNameNew ="000"+imageName
            elif len(imageName)==3:
                imageNameNew ="000"+imageName
            elif len(imageName)==4:
                imageNameNew ="00"+imageName
            elif len(imageName)==5:
                imageNameNew ="0"+imageName


                    
            Y_gaze[b] = np.array([gazePose0[i],gazePose1[i]])
            
            imageLeft = cv2.imread(datasetPath+"/"+folderName[i]+"/left/left_"+imageNameNew+"_rgb.png")
            imageRight = cv2.imread(datasetPath+"/"+folderName[i]+"/right/right_"+imageNameNew+"_rgb.png")
            
            resizedLeft = cv2.resize((imageLeft),(IMG_WIDTH,IMG_HEIGHT))/255
            resizedRight = cv2.resize((imageRight),(IMG_WIDTH,IMG_HEIGHT))/255
            

            
            image_batch[b]=resizedLeft
            image_batch1[b]=resizedRight


            
                
            
            i += 1

        batch_index=batch_index+1
        
#        image_batch, Y_batch0 = __augment_batch( np.uint8(image_batch*255), Y_batch0)

        yield [image_batch,image_batch1],[Y_gaze]
        



model = lightweight_model(input_shape=(48,80,3),learning_rate=0.0001)


MODEL_PATH='/home/ubuntu/sabari/gazeEstimation/model/'

model_checkpoint = ModelCheckpoint(MODEL_PATH+"weights--lightweight_model-light--single--48x80--channel-residual-{epoch:02d}-val_loss--{val_loss:.4f}---val_gaze_loss--{val_mean_squared_error:.4f}.hdf5",monitor='val_loss', 
                                   mode = 'min', save_best_only=True,save_weights_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode = 'min',factor=0.1, patience=5, min_lr=0.00001, verbose=1)

batch_size=16
results = model.fit_generator(generate_data(train,batch_size,True), validation_data=generate_data(valid,batch_size),steps_per_epoch=int(len(train)/batch_size),validation_steps=int(len(valid)/batch_size),
                              epochs=1000,callbacks=[model_checkpoint,reduce_lr], verbose=1)
#