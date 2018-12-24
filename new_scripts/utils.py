import numpy as np
import pandas as pd
import os
from os.path import join
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score


def create_lookup_df(img_path):
    vals = []
    trypos = []
    fpaths = []
    for root, dirs, files in os.walk(img_path):  
        for filename in files:
            filepath = join(root, filename)
            fpaths.append(filepath)
            
            if 'train' in filepath:
                vals.append(0)
            else: vals.append(1)
            
            if 'norm' in filepath:
                trypos.append(0)
            else: trypos.append(1)
                
    lookup_df = pd.DataFrame({'path': fpaths, 'validation': vals, 'trypo': trypos}) 
    lookup_df.reset_index(level=0, inplace=True)
    return lookup_df


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print(base_dir)
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

#read input images to generator from the pd.DataFrame
def get_generators(train_df, valid_df, input_s, core_train_gen, core_valid_gen):
    train_gen = flow_from_dataframe(core_train_gen, 
                              train_df,
                              path_col = 'path',
                              y_col = 'trypo',
                              target_size = (input_s, input_s),
                              color_mode = 'rgb',
                              batch_size = 32,
                              seed=8)

    valid_gen = flow_from_dataframe(core_valid_gen, 
                              valid_df,
                              path_col = 'path',
                              y_col = 'trypo',
                              target_size = (input_s, input_s),
                              color_mode = 'rgb',
                              batch_size = 32,
                              shuffle = False)
    
    return train_gen, valid_gen


def get_predictions_test(model, valid_df, input_s, core_valid_gen):
    test_gen = flow_from_dataframe(core_valid_gen, 
                              valid_df,
                              path_col = 'path',
                              y_col = 'trypo',
                              target_size = (input_s, input_s),
                              color_mode = 'rgb',
                              batch_size = 1,
                              shuffle = False)
    r = model.evaluate_generator(test_gen)
    print(r)
    predictions = model.predict_generator(test_gen)
    predictions = np.round([p[0] for p in predictions])
    
    labels = valid_df['trypo'].values
    acc = accuracy_score(labels, predictions)
    rep = classification_report(labels, predictions)
    print(acc, rep)
    return acc, rep

def divide_for_5CV(lookup_df):
    fold_ind = {}
    for i in range(5):
        fold_ind[i] = []
        
    norm_imgs = list(lookup_df[lookup_df.validation == 0][lookup_df.trypo == 0].index)
    trypo_imgs = list(lookup_df[lookup_df.validation == 0][lookup_df.trypo == 1].index)
    
    random_norm = random.sample(norm_imgs, len(norm_imgs))
    for i in range(len(norm_imgs)):
        fold_ind[i%5].append(random_norm[i])
        
    random_trypo = random.sample(trypo_imgs, len(trypo_imgs))
    for i in range(len(trypo_imgs)):
        fold_ind[i%5].append(random_trypo[i])        
    
    return fold_ind

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
