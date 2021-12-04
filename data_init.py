import os
import numpy as np
import cv2

dataSet = []

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def load_dir_images(dirname):
    for filename in os.listdir(dirname):
    	img_data = cv2.imread(os.path.join(dirname, filename))
    	dataSet.append((img_data, dirname[-1]))
        
def read_dataSet():
    for dirname in os.listdir('Dataset/'):
       load_dir_images ('Dataset/'+dirname)
    
def convert_labels_to_array(normalized_dataSet):
    for i in range ( len(normalized_dataSet) ):
        current_label = normalized_dataSet[i][1]
        current_img = normalized_dataSet[i][0]
        array = [0] * 10
        array[int(current_label)] = 1
        normalized_dataSet[i] = (current_img, list(array))
    return normalized_dataSet

def normalize_for_nn():
    normalized_dataSet = []
    read_dataSet()
    for img in dataSet:
        current_img = cv2.resize(img[0], (28,28), interpolation = cv2.INTER_AREA)
        current_img = rgb2gray(current_img)
        current_img = current_img.flatten() / 255.0
        current_img = np.round_( current_img , decimals=2 , out=None)
        current_img = [float(i) for i in current_img]
        normalized_dataSet.append((list(current_img), img[1]))
    normalized_dataSet = convert_labels_to_array(normalized_dataSet)
    return normalized_dataSet

def normalize_for_cnn():
    aveg_array = np.zeros(2352)
    flated_array = []
    sub_array = []
    normalized_dataSet = []
    read_dataSet()
    
    # Flaten data
    for img in dataSet:
        current_img = cv2.resize(img[0], (28,28), interpolation = cv2.INTER_AREA)
        current_img = current_img.flatten()
        flated_array.append((list(current_img), img[1]))
        aveg_array = np.add(current_img, aveg_array)
    aveg_array = aveg_array/len(dataSet)
    
    # Subtract average from data
    for img in flated_array:
        current_img = np.subtract(img[0], aveg_array) / 255.0
        current_img = np.around(current_img, decimals=3)
        sub_array.append((list(current_img), img[1]))    
    
    # Reshape Data again
    for img in sub_array:
        current_img = np.array(img[0]).reshape(28,28,3)
        normalized_dataSet.append((list(current_img), img[1]))    
    
    normalized_dataSet = convert_labels_to_array(normalized_dataSet)
    return normalized_dataSet















