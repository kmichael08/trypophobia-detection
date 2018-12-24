import numpy as np
import pandas as pd
import keras
import os
import random
import sys
import argparse
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from model import make_cut_vgg
from utils import create_lookup_df, flow_from_dataframe, get_generators, get_predictions_test, precision, recall


IMG_PATH = '/mnt/mydata'
TRAIN_NORM = 'train/norm'
TRAIN_TRYPO = 'train/trypo'
VALID_NORM = 'valid/norm'
VALID_TRYPO = 'valid/trypo'

os.environ['CUDA_VISIBLE_DEVICES'] = ('1')

if __name__ == '__main__':

    #parser = argparse.ArgumentParser()
    #parser.add_argument('layer_num', type=int, choices=[1,2,3,4], default=4)
    #parser.add_argument('downsample_factor', type=int, choices=[1,2,4], default=1)
    #parser.add_argument('save_name', type=str)
    #args=parser.parse_args()
    
    core_train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, 
                                             rotation_range=20, shear_range=0.2, zoom_range=0.2)
    core_valid_gen = ImageDataGenerator(rescale=1./255)
    lookup_df = create_lookup_df(IMG_PATH)
    
    EPOCH_NUM = 30
    #layers to cut after
    layer_names = ['block4_pool', 'block3_pool', 'block2_pool', 'block1_pool']
    downsample_factors = [2,4,4]
    
    for i in range(3):
        #downsampling the input image
        input_s = 224//downsample_factors[i]

        model = make_cut_vgg(layer_names[i], input_s, double_dense=True)
        model.compile(optimizer = optimizers.Adam(lr=1e-05, decay=0.01), loss="binary_crossentropy", metrics=["accuracy", precision, recall])

        train_df = lookup_df[lookup_df.validation == False]
        valid_df = lookup_df[lookup_df.validation == True]
        train_gen, valid_gen = get_generators(train_df, valid_df, input_s, core_train_gen, core_valid_gen)
        #sys.stdout = open('downsample_cut'+str(i)+'.txt', 'w')

        model.fit_generator(train_gen, 
                            steps_per_epoch=497,
                            validation_data=valid_gen,
                            validation_steps=50,
                            epochs=EPOCH_NUM, 
                            verbose=1)
        #model.save('vgg_'+args.save_name+'_test.h5')
        get_predictions_test(model, valid_df, input_s, core_valid_gen)