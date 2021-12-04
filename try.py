#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 01:01:54 2019

@author: amr
"""



import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

# find file paths
zero = glob.glob('Dataset/0/*')
one = glob.glob('Dataset/1/*')
two = glob.glob('Dataset/2/*')
three = glob.glob('Dataset/3/*')
four = glob.glob('Dataset/4/*')
five = glob.glob('Dataset/5/*')
six = glob.glob('Dataset/6/*')
seven = glob.glob('Dataset/7/*')
eight = glob.glob('Dataset/8/*')
nine = glob.glob('Dataset/9/*')

# total 1000 files for each category
print('Number of images per class:\n',len(zero),len(one),len(two),len(three),len(four),len(five),len(six),len(seven),len(eight),len(nine))