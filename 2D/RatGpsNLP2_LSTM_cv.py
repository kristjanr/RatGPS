# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 02:31:04 2022

@author: Renata Siimon
"""

from gensim.models.word2vec import Word2Vec  
from gensim.models import KeyedVectors 
from gensim.test.utils import datapath
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
# from tqdm import tqdm
from copy import deepcopy
import pickle as plk
import torch
import time
import re
import os
import math
import time
import torch

# import keras
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
#from wandb.keras import WandbCallback
#import wandb
from tensorflow.keras.losses import mean_squared_error

is_linux = True

# File separator:
if is_linux:
    s = "/"  # linux
else:
    s = "\\" # windows

path_to_data = "data" + s
# path_to_NLP = "NLP" + s
path_to_w2v = "word2vec" + s
path_to_results = "NLP_results" +s + "model3" + s


if torch.cuda.is_available():
    workers = torch.cuda.device_count() - 1  
    print('GPUs: ', torch.cuda.device_count())    
else:
    print('No GPUs...')

# -----------------------------------------------------

# LOAD DATA:
    
def read_data(fname):
    with open(fname, "r", encoding="UTF-8") as f:
        data = [line.rstrip().split(' ') for line in f.readlines()]
    df = pd.DataFrame(data = data)
    if len(df.columns) == 2:
        df.columns = ['x', 'y']
        for col in df.columns:
            df[col] = df[col].astype('float32')
    else:
        for col in df.columns:
            df[col] = df[col].astype('int64')        
    return df


# -----------------------------------------------------

# Sentences with no pauses, also update locations2 to leave out locations where there were no spikes:
# - allow no shuffling
# - allow merging 20ms intervals into larger ones (=define window size)
# -- then location is in the middle of the interval
# - also allow overlapping windows, i.e. window moves with certain step (=define step)
# - include param. "repetitions" - if True and neuron spiked multiple times, include it multiple times, otherwise include once all that spiked

def make_sents(dfx, locs, window_size, step, repetitions, do_shuffle): # window size - No of 20 ms intervals 
    sents = []
    spikes = [] # for comparison
    start, end = 0, window_size
    empty_windows = 0
    x_new, y_new = [], []
    x, y = locs['x'], locs['y']
    
    while end < len(dfx):
        
        # Sentences: 
        rows = dfx.iloc[start:end] # rows in window
        row = np.sum(rows) # all spikes of each neuron in window
        sent_words = []
        
        if np.sum(row)==0:
            empty_windows+=1
            sents.append(sent_words)
        else: # if there were any spikes at all
            for j, spike_count in enumerate(row):
                if spike_count!=0:
                    if repetitions==True:                    
                        sent_words+=[row.index[j] for x in range(spike_count)]                             
                    else:
                        sent_words.append(row.index[j])
            if do_shuffle==True:
                shuffle(sent_words)
            sents.append(sent_words)         
        
        # Spikes:
        spikes.append(row.tolist())    
            
        # Locations:
        if window_size==1:
            loc_x, loc_y = x[start], y[start]
        elif window_size%2==0: # even number
            loc_ind = int(start+(window_size)/2)
            loc_ind2 =  loc_ind-1
            loc_x = (x[loc_ind] + x[loc_ind2])/2
            loc_y = (y[loc_ind] + y[loc_ind2])/2
        else: # odd number
            loc_ind = int(start+(window_size-1)/2)
            loc_x, loc_y = x[loc_ind], y[loc_ind]
        x_new.append(loc_x)
        y_new.append(loc_y)
        start+=step
        end+=step
     
    locs2 = pd.DataFrame(data=[x_new, y_new]).T
    locs2.columns = ['x', 'y']   
    
    return [sents, locs2, empty_windows, spikes] 
   
    
# For analysis: how many empty and total sentences with different window size and step?
def analyse_windows(dfx, locations2, window_sizes, steps, repetitions):
    for i in range(len(steps)):
        window_size = window_sizes[i]
        step = steps[i]
        
        sents, locations3, empty_windows, spikes = make_sents(dfx, locations2, window_size=window_size, step=step, repetitions=repetitions, do_shuffle=False)
        sent_lens = pd.Series([len(s) for s in sents]).value_counts()
        print('\nwindow_size=', window_size, '(', window_size*20,'ms)', 'step=', step, '(', step*20,'ms)')
        print('empty windows: ', empty_windows, 'out of', len(sents), '(', round(100*empty_windows/(len(sents)), 2),'%), non-empty:', len(sents)-empty_windows)
        print(sent_lens)


# -----------------------------------------------------

# Weights for all neurons:
def calc_weights(dfx, locations2x, avg_loc):
    
    # Calculate centroid of each receptive field:
    neurons = dfx.columns.tolist()
    df_all = pd.concat([dfx, locations2x], axis=1)
    centroids_x, centroids_y = [], []
    centroids_y= []
    spike_times = []
    for neuron in neurons:
        d = df_all[df_all[neuron]>0]
        # d = df_all[df_all['42']>0]
        spike_times.append(len(d))
        if len(d)==0:  # if neuron didn't spike, assume it's centroid to be the default average location
            centroids_x.append(avg_loc[0])
            centroids_y.append(avg_loc[1])
        else:
            d = d[[neuron, 'x', 'y']]
            centroids_x.append(np.sum(d[neuron]*d['x'])/np.sum(d[neuron]))
            centroids_y.append(np.sum(d[neuron]*d['y'])/np.sum(d[neuron]))
    df_centroids = pd.DataFrame(data = [neurons, centroids_x, centroids_y, spike_times]).T
    df_centroids.columns = ['neuron', 'x', 'y', 'spike_times']
    df_centroids.index = neurons
    
    # Average distance of spike locations to centroid:
    means = []
    st_devs = []
    for neuron in neurons:
        d = df_all[df_all[neuron]>0]
        if len(d)==0: 
            means.append(45.0) # to have the missing neuron have very low weight
            st_devs.append(30.0)   # stdev is now not actually used...    
        else:
            d = d[[neuron, 'x', 'y']]
            # centroid = df_centroids.iloc[int(neuron)]
            centroid = df_centroids.loc[neuron]
            dists = np.sqrt((d['x']-centroid['x'])**2+(d['y']-centroid['y'])**2)
            avg_dist = np.mean(dists)  
            stdev_dists = np.std(dists)
            # plt.hist(dists, bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
            # plt.xlim(0,110)
            means.append(avg_dist)
            st_devs.append(stdev_dists) 
    dists_df = pd.DataFrame(data = [means, st_devs, spike_times]).T
    dists_df.columns = ['avg_dist', 'std_dist', 'spike_times']
    # plt.plot(sorted(dists_df['avg_dist']))
    
    dists_df.index = pd.Series(neurons).astype('int64')
    
    # Scale the distances:
    a = np.max(means)-means # inverse # 0- worst, 33.3- best
    # plt.plot(sorted(a)) # 
    a2 = (a/np.max(a))+0.1  # to avoid zero 
    
    # TODO: When assigning weights, take better into account how many times the neuron spiked - if it spiked only a few times, its weight should be lower (because then we have less confidence that in future it will spike in similar locations). 
    # - For now, just multiply weight by 0.9 if there were less than 10 spikes:
    b = pd.Series(spike_times)<10
    a2 = pd.Series(a2).where(-b, a2*0.9) 

    dists_df['weight'] = a2    
    dists_df['weight^3'] = a2**3 # to give more weight to more compact receptive fields
    dists_df['w+2*w^3']=dists_df['weight'] + dists_df['weight^3']*2 
    
    # plt.plot(sorted(dists_df['weight'] ))
    # plt.plot(sorted(dists_df['weight^3'] ))    
    # plt.plot(sorted(dists_df['w+2*w^3']))

    return dists_df

# dists_df = calc_weights(df, locations2)

# -----------------------------------------------------


# Final train and test:
def make_train_test(sents, locs2, spikes, train_index, test_index, dfx, locations2, win_size, step, rep, do_shuffle, remove_duplicates, fold):
        
    test_locs_tmp = locs2.iloc[test_index]
    train_locs_tmp = locs2.iloc[train_index]
    test_sents_tmp = pd.Series(sents).iloc[test_index].tolist()
    train_sents_tmp = pd.Series(sents).iloc[train_index].tolist()
    test_spikes_tmp = pd.Series(spikes).iloc[test_index].tolist()
    train_spikes_tmp =pd.Series(spikes).iloc[train_index].tolist()

    # Average location in train set (for predicting empty rows in test set):
    avg_loc = [np.mean(train_locs_tmp['x']), np.mean(train_locs_tmp['y'])]
    
    # Calculate weights for neurons, based on train set:
    # -Those indexes are based on 1/10 parts of the original dataframe (the one containing the sentences is already shorter, because we aggregated data in each window.
    split_size = int(len(dfx)/10)  # 5410
    if fold==0:
        start = split_size
        end =  len(dfx)
        dists_df = calc_weights(dfx.iloc[start:end], locations2.iloc[start:end], avg_loc)
    elif fold==9:
        start = 0
        end = len(dfx) - split_size
        dists_df = calc_weights(dfx.iloc[start:end], locations2.iloc[start:end], avg_loc)
    else: 
        start = 0
        end = fold*split_size
        start2 = fold*split_size + split_size
        end2 = len(dfx)
        dfx_part1 = dfx.iloc[start:end]
        dfx_part2 = dfx.iloc[start2:end2]
        locs_part1 = locations2.iloc[start:end]
        locs_part2 = locations2.iloc[start2:end2]    
        dfx_parts_all = dfx_part1.append(dfx_part2, sort = False) 
        locs_parts_all = locs_part1.append(locs_part2, sort = False) 
        dists_df = calc_weights(dfx_parts_all, locs_parts_all, avg_loc)
    
    # Separate empty rows and their locations from test set:
    # - (they will be put back before calculating RMSE)
    x_new, y_new, test_sents, test_spikes = [], [], [], []
    x_new_empty, y_new_empty = [], []
    x, y = test_locs_tmp['x'], test_locs_tmp['y']
    loc_ids = x.index.tolist()
    for i, sent in enumerate(test_sents_tmp):
        if len(sent)!=0:
            test_sents.append(sent)
            test_spikes.append(test_spikes_tmp[i])
            x_new.append(x[loc_ids[i]])
            y_new.append(y[loc_ids[i]])
        else:
            x_new_empty.append(x[loc_ids[i]])
            y_new_empty.append(y[loc_ids[i]])
    
    test_locs_empty = pd.DataFrame(data=[x_new_empty, y_new_empty]).T
    test_locs_empty.columns = ['x', 'y']
    test_locs = pd.DataFrame(data=[x_new, y_new]).T
    test_locs.columns = ['x', 'y']
        

    # If there is same sentence in consequtive windows, leave only one such sentence, and average the location (this occurs because of moving window: if 20ms intervals at the edges of the window were empty, same sentence was included several times)- ONLY MAKES SENCE IF "DO_SHUFFLE" = False. BUT ACTUALLY, SINCE WE SHUFFLE, THIS HAS BECOME USELESS...
    if remove_duplicates==True:
        x_new, y_new, train_sents2, train_spikes2 = [], [], [], []
        x, y = train_locs_tmp['x'], train_locs_tmp['y']
        prev_x, prev_y = [], []
        prev_sent = ['']
        for i, sent in enumerate(train_sents_tmp):
            x_current, y_current = x[i], y[i]
            if sent==prev_sent:
                prev_x.append(x_current)
                prev_y.append(y_current)                                   
            else:
                if i!=0:
                    train_sents2.append(prev_sent)
                    train_spikes2.append(train_sents_tmp[i-1])
                    x_new.append(np.mean(np.array(prev_x)))
                    y_new.append(np.mean(np.array(prev_y)))
                prev_x, prev_y = [x_current], [y_current]
                prev_sent = sent 
            if i==len(train_sents_tmp):
                train_sents2.append(sent)
                train_spikes2.append(train_sents_tmp[i])
                x_new.append(np.mean(np.array(prev_x)))
                y_new.append(np.mean(np.array(prev_y)))             
                
        train_locs2 = pd.DataFrame(data=[x_new, y_new]).T
        train_locs2.columns = ['x', 'y']
    else:
        train_locs2 = train_locs_tmp
        train_sents2 = train_sents_tmp
        train_spikes2 = train_spikes_tmp
    
    # Exclude remaining empty sentences from train:
    x_new, y_new, train_sents3, train_spikes3 = [], [], [], []
    x, y = train_locs2['x'].tolist(), train_locs2['y'].tolist()
    for i, sent in enumerate(train_sents2):
        if len(sent)!=0:
            train_sents3.append(sent)
            train_spikes3.append(train_spikes2[i])
            x_new.append(x[i])
            y_new.append(y[i])
    
    train_locs3 = pd.DataFrame(data=[x_new, y_new]).T
    train_locs3.columns = ['x', 'y']
    
    
    return [test_locs, test_locs_empty, test_sents, train_locs3, train_sents3, train_spikes3, test_spikes, avg_loc, dists_df]


# -----------------------------------------------------

class my_callback(CallbackAny2Vec): # to print loss after each epoch
    def __init__(self, model, epochs, model_path, s): 
        self.epoch = 1
        self.tot_epochs = epochs
        self.loss_previous_step=0
        self.model_path = model_path
        self.best_model = model
        self.best_epoch = 1
        self.best_loss = 1000000
        self.s = s
        
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            print("Training word2vec for " + str(self.tot_epochs) + " epochs...")
            current_loss = loss
        else:
            current_loss = loss-self.loss_previous_step # loss is cumulative
            
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_model  = model
            self.best_epoch = self.epoch
            # print("Epoch " + str(self.epoch)+ ": loss: " + str(round(current_loss, 6)))
            
        # Last epoch:
        if self.epoch == self.tot_epochs:
            self.best_model.save(self.model_path + self.s + "word2vec.model")  
            print("word2vec best epoch: " +  str(self.best_epoch) + ", loss: "+ str(round(self.best_loss, 6)))
            
        self.epoch+= 1
        self.loss_previous_step = loss  
         

def make_wordvec_model(train_sents, vec_len, window_size, skipgram, batch_size, epochs, s, path_to_w2v):

    model = Word2Vec(min_count=1, vector_size=vec_len, window=window_size, max_vocab_size=None, max_final_vocab=None, sg = skipgram,  compute_loss= True, batch_words=batch_size)

    # Build vocabulary:
    model.build_vocab(train_sents)
    # a = model.wv.key_to_index
    sent_counts = len(train_sents)
    
    # Train with callback:
    model.train(corpus_iterable = train_sents, total_examples=sent_counts, epochs=epochs, compute_loss=True, callbacks=[my_callback(model, epochs, path_to_w2v, s)]) 
        
    return model


# For standardizing the input (needed especially for LSTM, but also important for logistic regression; Random Forest doesn't care much):
def standardize_col(col_train, col_test):
    m = np.mean(col_train)
    stdev = np.std(col_train)
    col2_train = (col_train-m)/stdev
    col2_test = (col_test-m)/stdev   
    return [col2_train, col2_test]



def make_vecs(model, neurons, train_sents, test_sents, train_spikes, test_spikes, train_locs, test_locs, dists_df, use_weights):      

    weight_col = dists_df['w+2*w^3'] # 'w+2*w^3', '5*(w+w^3)' # 'w+w^2+w^3'  'w+w^3' #  #  '3*(w+w^3)' # '5*(w+w^3)'  # 'w+5*w^3' # 'w+3*w^3' #'7*(w+w^3)' #  # '10*(w+w^3)' # '0.5*w+2*w^3',
     
    #-------------------------------------------------    
    # Sentences:
    
    # Average vectors of each sentence:
    train_vecs, test_vecs = [], [] # both train and test

    weight_col.index = weight_col.index.astype('str')
    for sent in train_sents:
        # sent = train_sents[0]
        vecs = [model.wv[code] for code in sent]
        
        # simple average:
        if use_weights==False: # simple average
            train_vecs.append(np.mean(np.array(vecs), axis = 0)) 
            
        # weighted average:
        else: 
            weights = [weight_col.loc[sent[x]] for x in range(len(sent))]
            vecs = vecs*np.array(weights).reshape(len(weights),1)
            train_vecs.append(np.mean(vecs, axis=0))
            
    for sent in test_sents:
        vecs = [model.wv[code] for code in sent]
        
        # simple average:
        if use_weights==False: # simple average
            test_vecs.append(np.mean(np.array(vecs), axis = 0)) 
            
        # weighted average:
        else: 
            weights = [weight_col.loc[sent[x]] for x in range(len(sent))]
            vecs = vecs*np.array(weights).reshape(len(weights),1)
            test_vecs.append(np.mean(vecs, axis=0))
        
    df_train = pd.DataFrame(data=train_vecs)
    df_test = pd.DataFrame(data=test_vecs)

    # Normalize:
    for col in df_train.columns.tolist():
        train_col, test_col = standardize_col(df_train[col], df_test[col])
        df_train[col] = train_col
        df_test[col] = test_col 

    #-------------------------------------------------    
    # Locations:
    train_y = train_locs
    test_y = test_locs

    #-------------------------------------------------
    
    # Spikes:
    df_train_spikes = pd.DataFrame(data=train_spikes)
    df_test_spikes = pd.DataFrame(data=test_spikes)

    # Not sure if multiplying each feature by some weight gives any effect...
    if use_weights==True:
        df_train_spikes = np.true_divide(df_train_spikes, weight_col.tolist()) 
        df_test_spikes = np.true_divide(df_test_spikes, weight_col.tolist()) 
    
    # Normalize:
    for col in df_train_spikes.columns.tolist():
        train_col, test_col = standardize_col(df_train_spikes[col], df_test_spikes[col])
        df_train_spikes[col] = train_col
        df_test_spikes[col] = test_col 
    
    
    return [train_vecs, test_vecs, df_train, df_test, train_y, test_y, df_train_spikes, df_test_spikes]


def get_predictions(reg_model, df_train2, train_y3, df_test2, test_y3, avg_loc, test_locs_empty, test_sents, lstm_seqlen):
    
     # "<class 'tensorflow.python.keras.engine.functional.Functional'>"
    if "tensorflow" in str(type(reg_model)):  # Reshape test
        test_features = np.array(df_test2)
        df_test2, test_y3 = sliding_window(test_features, test_y3, lstm_seqlen)
        preds = reg_model.predict(df_test2)
    else:     
        reg_model = reg_model.fit(df_train2, train_y3)
        preds = reg_model.predict(df_test2)
    
    # Predictions (also add back intervals with no spikes, for which we always predict average location of rat in train set):
    preds_df = pd.DataFrame(preds)
    preds_avg_x = [avg_loc[0] for x in range(len(test_locs_empty))]
    preds_avg_y = [avg_loc[1] for x in range(len(test_locs_empty))]
    empty_preds_df = pd.DataFrame(data = [preds_avg_x, preds_avg_y]).T
    preds_df = preds_df.append(empty_preds_df, ignore_index=True, sort = False) 
    preds_df.columns=['x', 'y']
    
    # Actuals for emptys:
    emptys_count = len(test_locs_empty)
    if emptys_count!=0:
        test_y3 = test_y3.append(test_locs_empty, ignore_index=True, sort = False)
        
    if len(test_sents)!=0:  # only for sentences (not spikes)
        test_sents2 = deepcopy(test_sents)
        if "tensorflow" in str(type(reg_model)):
            test_sents2 = test_sents2[seqlen-1:]
        if emptys_count!=0:
            for i in range(emptys_count):
                test_sents2.append([])

    
    # Distance between predicted and actual location:
    tmp_preds = preds_df.reset_index(drop=True)
    tmp_test_y = test_y3.reset_index(drop=True)
    dists = np.sqrt((tmp_test_y['x'] - tmp_preds['x'])**2 + (tmp_test_y['y'] -tmp_preds['y'])**2)
    avg_dist_te = np.mean(dists)   
    # print('test:', avg_dist_te, end = ', ')
    median_dist_te = np.median(dists)
    
    
    # Predictions for train: (doesn't include possible empty rows)
    if "tensorflow" in str(type(reg_model)):  # Reshape test
        tr_features = np.array(df_train2)
        df_train2, train_y3 = sliding_window(tr_features, train_y3, lstm_seqlen)
        
    preds2 = reg_model.predict(df_train2)
    preds_df2 = pd.DataFrame(preds2)
    preds_df2.columns=['x', 'y']
    tmp_preds2 = preds_df2.reset_index(drop=True)
    tmp_train_y3 = train_y3.reset_index(drop=True)
    
    dists2 = np.sqrt((tmp_train_y3['x'] - tmp_preds2['x'])**2 + (tmp_train_y3['y'] - tmp_preds2['y'])**2)
        
    avg_dist_tr = np.mean(dists2)   
    # print('train:', avg_dist_tr)
    median_dist_tr = np.median(dists2)
    
    # Results as df:
    df_preds = pd.DataFrame(data = [test_y3['x'].tolist(), test_y3['y'].tolist(), preds_df['x'].tolist(), preds_df['y'].tolist(), dists.tolist()]).T
    df_preds.columns = ['x', 'y', 'pred_x', 'pred_y', 'dist']
        
    # Sentence lengths - only for sentences (not spikes):
    if len(test_sents)!=0:
        sent_lengths = [len(sent) for sent in test_sents2]
        df_preds['sent']= test_sents2  
        df_preds['length']= sent_lengths
        df_preds = df_preds[['sent', 'length', 'x', 'y', 'pred_x', 'pred_y', 'dist']]
    
    return [df_preds, avg_dist_te, avg_dist_tr, median_dist_te, median_dist_tr]

# ----------------------------------------------------------

# LSTM:

def sliding_window(X, y, seqlen):
    # X - matrix where each row corresponds to a spike count vector of length nr_of_neurons
    # y - rat positions at the center of those spike count windows
    # seqlen - length of sequences we want to get out of this function

    Xs = []
    for i in range(seqlen): #0...99        
        #100-99-1 >0
        if seqlen - i - 1 > 0: #not the last piece
            # take slices from 0 to -99, 1 to -98, ...,  98 to -1.
            Xs.append(X[i:-(seqlen - i - 1), np.newaxis, ...])
        else:  # cannot ask X[99:-0], so special case goes here
            #print("last piece to add")
            Xs.append(X[i:, np.newaxis, ...])

    # we have seqlen(=100) slices each shifted in time. join them to get sequences of len 100
    X = np.concatenate(Xs, axis=1)
    y = y[seqlen - 1:]  # the positions are taken at the last timestep (from 99 to end)
    print(f"After sliding window: {(X.shape, y.shape)}")

    return X, y


def get_early_stopping_config(patience=0):
  return keras.callbacks.EarlyStopping(patience=patience, verbose=1)

def get_lr_config(lr=0.001, lr_factor=0.1, lr_epochs=None):
  def lr_scheduler(epoch):
    new_lr = lr * lr_factor ** int(epoch / lr_epochs)
    print(f"Epoch {epoch}: learning rate {new_lr}")
    return new_lr
  return keras.callbacks.LearningRateScheduler(lr_scheduler)

def get_savepoints_config(filepath, save_best_model_only=False):
  return keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_best_model_only=save_best_model_only)


def split_ndarray(df, train_ratio):
  idx = math.floor(len(df) * train_ratio)
  out = df[:idx], df[idx:]
  return out

def mse(y, t, axis=-1):
    return np.square(y - t).mean(axis=axis).mean()

def mean_distance(y, t, axis=-1):
    return np.mean(np.sqrt(np.sum((y - t) ** 2, axis=axis)))

def median_distance(y, t, axis=-1):
    return np.median(np.sqrt(np.sum((y - t) ** 2, axis=axis)))


def eval_model(model, X, y):  # enne: eval()
  pred_y = model.predict(X)
  if type(y) == pd.core.frame.DataFrame:
    y = y.to_numpy()
  err = mse(pred_y, y)
  dist = mean_distance(pred_y, y)
  median_dist = median_distance(pred_y, y)
  return err, dist, median_dist


def split_df(df, train_ratio):
  idx = math.floor(len(df) * train_ratio)
  return df[:idx], df[idx:]


def createLSTM(seqlen, num_features, dropout_ratio, hidden_nodes=256): # 512 # 256
    #  # hidden_nodes=1024 - Kristjanil: 1024 ; artiklis 512; ise proovides tundus 256 ok
    num_outputs = 2

    x = keras.layers.Input(shape=(seqlen, num_features), name="Input")
    h = x

    # first LSTM layer with dropout
    h = keras.layers.LSTM(hidden_nodes, input_shape=(seqlen, num_features), return_sequences=True, name="firstLstmBlock")(h) # 
    h = keras.layers.Dropout(dropout_ratio, name="firstLstmDropout")(h)

    # second LSTM layer with dropout
    h = keras.layers.LSTM(hidden_nodes, name="secondLstmBlock")(h) # return_sequences=True
    h = keras.layers.Dropout(dropout_ratio, name="secondLstmDropout")(h)

    # finally, add dense layer which acts as output layer
    y = keras.layers.Dense(num_outputs, name="Output")(h)

    opt = keras.optimizers.RMSprop(learning_rate=0.001) # centered = True
    #  no effect: momentum = 0.3 (default 0.0)
    # https://keras.io/api/optimizers/rmsprop/
    # optimizer "rmsprop": default learning rate: 0.001 (same as in article)

    model = keras.models.Model(inputs=x, outputs=y, name="RatLSTM") # suppress progress bar - for slurm
    model.compile(loss=mean_squared_error, optimizer=opt) # optimizer="rmsprop"

    return model



def test_lstm_nlp(input_features, locations, seqlen, train_ratio, epochs, dropout_ratio):
  input_features = np.array(input_features)
  train_feats, test_feats = split_ndarray(input_features, train_ratio)
  train_y, test_y = split_df(locations, train_ratio)

  train_X, train_y = sliding_window(train_feats, train_y, seqlen)
  test_X, test_y = sliding_window(test_feats, test_y, seqlen)
  # num_features=train_X.shape[2] # 125

  model = createLSTM(seqlen=seqlen, num_features=train_X.shape[2], dropout_ratio=dropout_ratio)
  
  print(model.summary())
  print(train_X.shape)
  print(train_y.shape)
  print(test_X.shape)
  print(test_y.shape)

  #callbacks = [WandbCallback()]
  epochs = epochs
  batch_size = 64 #32 
  config = {
    "epochs": epochs, 
    "batch_size": batch_size,
    "model-type": "original",
    "data": "mean of word2vectors",
    "seqlen": seqlen,
    "train_X.shape": train_X.shape,
    "train_y.shape": train_y.shape,
    "test_X.shape": test_X.shape,
    "test_y.shape": test_y.shape,
    
  }
  # wandb.init(project="LSTM", entity="compneuro", config=config)
  history = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_y), shuffle=False, verbose = 2) # callbacks=callbacks, verbose = 2 
  
  terr, tdist, tmediandist = eval_model(model, train_X, train_y)
  verr, vdist, vmediandist = eval_model(model, test_X, test_y)
  print('train mse = %g, validation mse = %g' % (terr, verr))
  print('train mean dist = %g, validation mean dist = %g' % (tdist, vdist))
  print('train median dist = %g, validation median dist = %g' % (tmediandist, vmediandist))
  
  return [model, history]

def plot_losses(history, epochs, seqlen, dropout):
    name = "ep=" + str(epochs) + ", seqlen=" + str(seqlen) + ", drop=" + str(dropout) 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylim(0,500)
    plt.title(name)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
        
    print('\nRMSEs of models with spikes (fold=', fold, '):')
    
    # For LSTM models, train mean and train median error should be taken from model output, not from output of get_predictions(), because the latter uses whole original train set (before separating validation set for LSTM), and therefore is less accurate (bec. validation set was not used for training). 

#-----------------------------------------------------------

# LOAD DATA:
    
# 5410 steps:
locations = read_data(path_to_data + "R2192_1x200_at5_step200_bin100-RAW_pos.dat")

# 54100 steps:
locations2 = pd.read_csv(path_to_data + "R2192_20ms_speed_direction.csv", encoding = "UTF-8") 
locations2 =locations2.drop(columns= ['speed', 'direction', 'direction_disp'])

# rescale locations2 so that they are the same as locations (1mx1m):
x_factor = locations.iloc[0]['x']/np.mean(locations2[:10]['x'])
y_factor = locations.iloc[0]['y']/np.mean(locations2[:10]['y'])
locations2['x'] = locations2['x']*x_factor
locations2['y'] = locations2['y']*y_factor

# Spikes:
fn = "R2192_20ms_63_neurons.csv" # 63 rows, 54100 cols
df =  pd.read_csv(path_to_data + fn, encoding = "UTF-8", header=None) 
df = df.T

# Neuron ids:
neurons = [str(x) for x in range(63)]
df.columns = neurons

#-----------------------------------------------------------

# CROSS-VALIDATION:
    
window_size = 60 #60 # 50
step =  7 # 7 # 10 # 3 # 10
do_shuffle= True # TRUE!  
repetitions = True # TRUE!
rm_duplicates = False # # FALSE! (doesn't make sence if do_shuffle=True)
data = df #
w2v_vec_len= 125 # 250
w2v_win_size = 7 # 6 # 5 #10 #10 # 5
w2v_skipgram=0
w2v_batch_size= 500 #500 # 500 - ok with non-augmented train data
w2v_epochs= 600 #600 # 600- good with non-augmented train data
use_weights = True # use weighted average instead of simple average when combining neuron vectors into sentence


# window_size = 10 -> 200ms
# window_size = 60 -> 1200ms  

# Sentences, spikes and locations:
window_data = make_sents(df, locations2, window_size, step,  repetitions, do_shuffle)
sents, locs2, spikes = window_data[0], window_data[1], window_data[3]


#10-fold cv rmse-s:
rmses_lstm_spikes = []   #[[fold1_te, fold1_tr], [fold2_te, fold2_tr], ...]
rmses_lstm_sent = []
medians_lstm_sent = []
medians_lstm_spikes = []

# Predictions for test:
pred_lstm_spikes, pred_lstm_sents = [], []
 #[fold1_df_pred, fold2_df_pred ...] # not strictly needed, just interesting to compare what was predicted...
 
# Input parameters for LSTM:
epochs = 50 #50 # in article: 50
seqlen = 50 # 30  # in article: 100
dropout = 0.5 # # in article: 0.5

# note: check also "hidden_nodes" in LSTM model definition above (in article, hidden_nodes=512 was used)

print("Epochs: " + str(epochs))
print("seqlen:" +  str(seqlen))
print("dropout:" +  str(dropout))

def end_time(start, fold, categ):
    end = time.perf_counter()
    minutes = (end - start)/60
    hours = str(round(minutes/60, 3))
    minutes = str(round(minutes, 3))
    print("Fold "+ str(fold) + " (" + categ + ") done in: " + minutes + " minutes (" + hours + " hours)")

kf = KFold(n_splits=10, shuffle = False)
kfold  = kf.split(sents)  

w2v_numjobs = 6


# for fold in tqdm(range(0,10)):
for fold in range(0, 10):

    start = time.perf_counter()
    
    # Indexes of this fold:
    train_index, test_index = next(kfold) # kfold is generator object
    
    print("======================================================")
    print('\nFOLD = ', fold, ':')
    print("======================================================")
    
    # if fold!=9:   # for testing
    #     continue
    # if fold==1:
    #    break
    
    #PREPARE DATA FOR THIS FOLD: 
        
    print('preparing data...')
    result2 = make_train_test(sents, locs2, spikes, train_index, test_index, data, locations2, window_size, step, repetitions, do_shuffle, rm_duplicates, fold)
    test_locs, test_locs_empty, test_sents = result2[0], result2[1], result2[2]
    train_locs, train_sents = result2[3], result2[4]
    train_spikes, test_spikes = result2[5], result2[6]
    avg_loc, dists_df = result2[7], result2[8]
    #print('Min spikes of neuron:', int(np.min(dists_df['spike_times'])))
    
    # Make word2vec model: ONLY SENTS:
    model = make_wordvec_model(train_sents, w2v_vec_len, w2v_win_size, w2v_skipgram, w2v_batch_size, w2v_epochs, s, path_to_w2v) # last model (might not be best)
    model = Word2Vec.load(path_to_w2v + s + "word2vec.model") # laod best model      
    
    # Make vectors (only for sentences), and prepare (weighted) train/test dfs (for sents and spikes):
    result = make_vecs(model, neurons, train_sents, test_sents, train_spikes, test_spikes, train_locs, test_locs, dists_df, use_weights)
    train_vecs, test_vecs = result[0], result[1]
    df_train, df_test= result[2], result[3]
    df_train_spikes, df_test_spikes= result[6], result[7]  # pmst: trainX, testX
    train_y, test_y = result[4], result[5]
    
    end_time(start, fold, "prep data")
    
    # -------------------------------------------
    
    # MODEL FOR SENTENCES:
    
    start = time.perf_counter()
    
    model, history = test_lstm_nlp(input_features=df_train, locations=train_y, seqlen=seqlen, train_ratio=0.99, epochs = epochs, dropout_ratio=dropout)
   
    # Predictions for test:
    df_preds, rmse_te, rmse_tr, median_te, median_tr = get_predictions(model, df_train, train_y, df_test, test_y, avg_loc, test_locs_empty, test_sents, lstm_seqlen=seqlen)
    print('\nErrors of models with sentences (fold=', fold, '):')
    print('mean: test: ', rmse_te, 'train:', rmse_tr)
    print('median: test: ', median_te, 'train:', median_tr)
    
    rmses_lstm_sent.append([rmse_te, rmse_tr])
    medians_lstm_sent.append([median_te, median_tr])
    pred_lstm_sents.append(df_preds)
    
    # Save:
    fname = "LSTM_fold_" + str(fold) + "_results_sents.plk"
    with open(path_to_results + fname, "wb") as fout:
        plk.dump([df_preds, rmse_te, rmse_tr, median_te, median_tr], fout) 
    
    # # # For loading, use:
    # fname = "LSTM_fold_" + str(fold) + "_results_sents.plk"
    # with open(path_to_results + fname, "rb") as fin:
    #     loaded = plk.load(fin)
    #     df_preds =loaded[0] 
    #     rmse_te = loaded[1]
    #     rmse_tr =loaded[2] 
    #     median_te = loaded[3]        
    #     median_tr = loaded[4]  

    end_time(start, fold, "sents")
    print('-------------------------------------------\n')
    
    # MODEL FOR SPIKES:
        
    start = time.perf_counter()
    
    model, history = test_lstm_nlp(input_features=df_train_spikes, locations=train_y, seqlen=seqlen, train_ratio=0.99, epochs = epochs, dropout_ratio=dropout)
   
    # Predictions for test:
    df_preds, rmse_te, rmse_tr, median_te, median_tr = get_predictions(model, df_train_spikes, train_y, df_test_spikes, test_y, avg_loc, test_locs_empty, [], lstm_seqlen=seqlen)
    print('\nErrors of models with spikes (fold=', fold, '):')
    print('mean: test: ', rmse_te, 'train:', rmse_tr)
    print('median: test: ', median_te, 'train:', median_tr)
    
    rmses_lstm_spikes.append([rmse_te, rmse_tr])
    medians_lstm_spikes.append([median_te, median_tr])
    pred_lstm_spikes.append(df_preds)    
        
        
    # Save:
    fname = "LSTM_fold_" + str(fold) + "_results_spikes.plk"
    with open(path_to_results + fname, "wb") as fout:
        plk.dump([df_preds, rmse_te, rmse_tr, median_te, median_tr], fout) 
    
    # For loading, use:
    # fname = "LSTM_fold_" + str(fold) + "_results_spikes.plk"
    # with open(path_to_results + fname, "rb") as fin:
    #     loaded = plk.load(fin)
    #     df_preds =loaded[0] 
    #     rmse_te = loaded[1]
    #     rmse_tr =loaded[2] 
    #     median_te = loaded[3]        
    #     median_tr = loaded[4]  
    
    end_time(start, fold, "spikes")


# -------------------------------------------------
   
# All results:
rmses = [rmses_lstm_spikes, rmses_lstm_sent]
medians = [medians_lstm_spikes, medians_lstm_sent]
cols = ['LSTM_spikes', 'LSTM_sents']
df_rmse_te = pd.DataFrame()
df_rmse_tr = pd.DataFrame()
df_median_te = pd.DataFrame()
df_median_tr = pd.DataFrame()

for i, model_rmses in enumerate(rmses):
    model_rmses = pd.DataFrame(data = model_rmses)
    df_rmse_te[cols[i]] = model_rmses[0]
    df_rmse_tr[cols[i]] = model_rmses[1]    
 
for i, model_medians in enumerate(medians):
    model_medians = pd.DataFrame(data = model_medians)
    df_median_te[cols[i]] = model_medians[0]
    df_median_tr[cols[i]] = model_medians[1]        
 

print("\n------------------------------------")
print("FINAL RESULTS:")
print("\nMEAN DISTANCES IN TEST SET (cm):")
print(df_rmse_te)

print("\nMEDIAN DISTANCES IN TEST SET (cm):")
print(df_median_te)

print("\nAvgerages over 10 folds:" + str())
tmp = pd.DataFrame(data = [np.mean(df_rmse_te), np.mean(df_median_te), np.std(df_rmse_te), np.std(df_median_te)]).T
tmp.columns = ['Mean error', 'Median error', 'st.dev. of mean', 'st.dev of median']
print(tmp)




# -------------------------------------------------

# COMBINE ALL PREDICTION DATAFRAMES:

def make_single_preds_df(cols, is_spikes, dfs_orig):
    dfs_list = []
    for i, models_preds in enumerate(dfs_orig): # model
        for k in range(10): # fold
            fold_preds = models_preds[k]
            if i==0:
                if is_spikes==True:
                    df_preds_fold = fold_preds[['x', 'y']]
                else:
                    df_preds_fold = fold_preds[['sent', 'length', 'x', 'y']]
            else:
                df_preds_fold = dfs_list[k]
            model_name = cols[i].split('_')[0]
            df_preds_fold[model_name + "_x"] = fold_preds['pred_x']
            df_preds_fold[model_name + "_y"] = fold_preds['pred_y']
            df_preds_fold[model_name + "_dist"] = fold_preds['dist']  
            if i==0:
                dfs_list.append(df_preds_fold )
            else:
                dfs_list[k] = df_preds_fold      
    return dfs_list

pred_dfs_spikes = [pred_lstm_spikes]
pred_dfs_sents =  [pred_lstm_sents]
cols_spikes = ['LSTM_spikes']
cols_sents = ['LSTM_sents']

dfs_pred_spikes = make_single_preds_df(cols_spikes, True, pred_dfs_spikes)
dfs_pred_sents = make_single_preds_df(cols_sents, False, pred_dfs_sents)


#----------------------------

# SAVE RESULTS:

with open(path_to_results + "LSTM_dfs_preds_folds.plk", "wb") as fout:
    plk.dump([dfs_pred_spikes, dfs_pred_sents], fout) 

df_rmse_te.to_csv(path_to_results + "LSTM_rmses_test.csv", sep=",", index=False, encoding="utf-8") 
df_rmse_tr.to_csv(path_to_results + "LSTM_rmses_train.csv", sep=",", index=False, encoding="utf-8") 
df_median_te.to_csv(path_to_results + "LSTM_medians_test.csv", sep=",", index=False, encoding="utf-8") 
df_median_tr.to_csv(path_to_results + "LSTM_medians_train.csv", sep=",", index=False, encoding="utf-8") 


#----------------------------

# LOAD RESULTS:
# df_rmse_te = pd.read_csv(path_to_results + "rmses_test.csv", sep=",", encoding = "UTF-8") 
# df_rmse_tr = pd.read_csv(path_to_results + "rmses_train.csv", sep=",", encoding = "UTF-8") 

# with open(path_to_results + "dfs_preds_folds.plk", "rb") as fin:
#     loaded = plk.load(fin)
#     dfs_pred_spikes =loaded[0] 
#     dfs_pred_sents = loaded[1]

#-----------------------------------------------------------

# ## ANALYSE:
# window_sizes = [30, 30, 30]
# steps = [5, 10, 15]
# analyse_windows(df, locations2, window_sizes, steps, repetitions=False)

# window_sizes = [30, 40, 50]
# steps = [15, 20, 25]
# analyse_windows(df, locations2, window_sizes, steps, repetitions=False)

# window_sizes = [60, 70, 80]
# steps = [30, 35, 40]
# analyse_windows(df, locations2, window_sizes, steps, repetitions=False)
