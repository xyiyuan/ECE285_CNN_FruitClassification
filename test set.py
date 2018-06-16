import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import os

#Test set
print("Loading test set...")

test_labels=[]
test_images=[]
for dir_path in glob.glob("/datasets/ee285s-public/fruits-360/Validation/*"):
    label=dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(dir_path,"*.jpg")):
        test_labels.append(label)
        image=cv2.imread(image_path,cv2.IMREAD_COLOR)
        test_images.append(image)

test_labels=np.array(test_labels)
test_images=np.array(test_images)
print('Done')

label_to_id_dict = {v:k for k,v in enumerate(np.unique(test_labels))}
id_to_label_dict = {v:k for k,v in label_to_id_dict.items()}

#Split test set into val/test set
print("Splitting into val/test set...")

idx=np.random.permutation(range(9673))

test_images,test_labels=test_images[idx[4841:9673]],test_labels[idx[4841:9673]]

print('Done')

#Normalize
print("Normalization...")

test_images=test_images/255.0

print("Done")

test_labels_ids=np.array([label_to_id_dict[i] for i in test_labels])

#Labels one-hot encode
print("Converting to one-hot code...")

hot_encode_test=np.zeros((test_labels_ids.max()+1,test_labels_ids.size))
hot_encode_test[test_labels_ids,np.arange(0,test_labels_ids.size)]=1

test_labels=hot_encode_test.T

print("Done")

X_test=test_images
Y_test=test_labels

print("Finished!")