#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:33:42 2020

@author: wuzongze


Modified on Wed Jul 20 11:29 2022

@author Susmit-A
"""

import os
import sys
sys.path.append('..')
import dnnlib.tflib as tflib
import numpy as np
from PIL import Image
import pandas as pd

import matplotlib.pyplot as plt
import argparse
import glob
from tqdm import tqdm
import pickle as pkl

def convert_images_from_uint8(images, drange=[-1,1], nhwc_to_nchw=False):
    """Convert a minibatch of images from uint8 to float32 with configurable dynamic range.
    Can be used as an input transformation for Network.run().
    """
    if nhwc_to_nchw:
        imgs_roll=np.rollaxis(images, 3, 1)
    return imgs_roll/ 255 *(drange[1] - drange[0])+ drange[0]

#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict pose of object')
    parser.add_argument('--img_path',type=str,help='path to image folder')
    parser.add_argument('--save_path',type=str,help='path to save attribute file') 
    parser.add_argument('--classifier_path',default='./attr_models',type=str,help='path to a folder of classifers') 
    
    opt = parser.parse_args()
    
    img_path=opt.img_path
    save_path=opt.save_path
    classifer_path=opt.classifier_path
    
    imgs = glob.glob(os.path.join(opt.img_path, '*.png'))
    names_tmp=os.listdir(classifer_path)
    names=[]
    for name in names_tmp:
        if 'celebahq-classifier' in name:
            names.append(name)
    names.sort()
    
    tflib.init_tf()
    classifiers = []
    print("Initializing models")
    for name in tqdm(names):
        tmp=os.path.join(classifer_path,name)
        with open(tmp, 'rb') as f:
            classifier = pkl.load(f)
        classifiers.append((name, classifier))
    
    results={}
    for file in tqdm(imgs):
        img_name = file.split('/')[-1]
        results[img_name] = {}
        for (name, model) in classifiers:
            img = np.array(Image.open(file)) 
            tmp_imgs = np.stack(np.split(img, 11, axis=1), axis=0)
            tmp_imgs=convert_images_from_uint8(tmp_imgs, drange=[-1,1], nhwc_to_nchw=True)
            tmp = model.run(tmp_imgs, None)
            
            tmp1 = tmp.reshape(-1) 
            results[img_name][name] = tmp1
        
    with open(opt.save_path, 'wb') as f:
        pkl.dump(results, f)

