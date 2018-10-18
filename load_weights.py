# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:16:06 2018

@author: Guest NGN
"""

import pickle
import argparse
import tensorflow as tf

def load_from_pkl(file_path):
    args_loaded = {}
    auxs_loaded ={}
    
    with open(file_path, 'rb') as fopen:
        blobs = pickle.load(fopen, encoding='latin-1')['blobs']
    
    print("len of blobs %d" % (len(blobs)))
    
    for k, v in sorted(blobs.items()):
        print(k)
        if len(v.shape) == 2:
            v = tf.transpose(v)
            print(v.shape)
        elif len(v.shape) == 5:
            v = tf.transpose(v, perm=[2,3,4,1,0])
            print(v.shape)
        else:
            print(v.shape)
    
    #conv1_w = blobs['conv1_middle_w']
    #print(conv1_w.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, help='saved weights file')
    
    args = parser.parse_args()
    
    load_from_pkl(args.pkl)