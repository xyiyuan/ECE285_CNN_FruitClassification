import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import os

#Training set
print("Loading training set...")

train_labels=[]
train_images=[]
for dir_path in glob.glob("/datasets/ee285s-public/fruits-360/Training/*"):
    label=dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(dir_path,"*.jpg")):
        train_labels.append(label)
        image=cv2.imread(image_path,flags=cv2.IMREAD_COLOR)
        train_images.append(image)

train_labels=np.array(train_labels)
train_images=np.array(train_images)        

print("Done")

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

#Split test set into val/test set
print("Splitting into val/test set...")
idx=np.random.permutation(range(9673))
#Val
val_images,val_labels=test_images[idx[0:4841]],test_labels[idx[0:4841]]
#Test
test_images,test_labels=test_images[idx[4841:9673]],test_labels[idx[4841:9673]]

print('Done')

#Normalize
print("Normalization...")

#Train
train_images=train_images/255
#Val
val_images=val_images/255
#Test
test_images=test_images/255

print("Done")

label_to_id_dict = {v:k for k,v in enumerate(np.unique(train_labels))}
id_to_label_dict = {v:k for k,v in label_to_id_dict.items()}

train_labels_ids=np.array([label_to_id_dict[i] for i in train_labels])
val_labels_ids=np.array([label_to_id_dict[i] for i in val_labels])
test_labels_ids=np.array([label_to_id_dict[i] for i in test_labels])

#Labels one-hot encode
print("Converting to one-hot code...")
#Train
hot_encode_train=np.zeros((train_labels_ids.max()+1,train_labels_ids.size))
hot_encode_train[train_labels_ids,np.arange(0,train_labels_ids.size)]=1
#Val
hot_encode_val=np.zeros((val_labels_ids.max()+1,val_labels_ids.size))
hot_encode_val[val_labels_ids,np.arange(0,val_labels_ids.size)]=1
#Test
hot_encode_test=np.zeros((test_labels_ids.max()+1,test_labels_ids.size))
hot_encode_test[test_labels_ids,np.arange(0,test_labels_ids.size)]=1

train_labels=hot_encode_train.T
val_labels=hot_encode_val.T
test_labels=hot_encode_test.T

print("Done")

X_train=train_images
Y_train=train_labels
X_val=val_images
Y_val=val_labels
X_test=test_images
Y_test=test_labels

def get_mask_coord(imshape):
    vertices = np.array([[(0.09 * imshape[1], 0.99 * imshape[0]), 
                          (0.43 * imshape[1], 0.32 * imshape[0]), 
                          (0.56 * imshape[1], 0.32 * imshape[0]),
                          (0.85 * imshape[1], 0.99 * imshape[0])]], dtype = np.int32)
    return vertices

def get_perspective_matrices(X_img):
    offset = 15
    img_size = (X_img.shape[1], X_img.shape[0])

    # Estimate the coordinates of object of interest inside the image.
    src = np.float32(get_mask_coord(X_img.shape))
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0], 
                      [img_size[0] - offset, img_size[1]]])
    
    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    return perspective_matrix

def perspective_transform(X_img):
    # Doing only for one type of example
    perspective_matrix = get_perspective_matrices(X_img)
    warped_img = cv2.warpPerspective(X_img, perspective_matrix,
                                     (X_img.shape[1], X_img.shape[0]),
                                     flags = cv2.INTER_LINEAR)
    return warped_img

print("Generating perspective images...")

train_perspective_images=[]
for image in train_images:
    perspective_image = perspective_transform(image)
    train_perspective_images.append(perspective_image)
    
train_perspective_images = np.array(train_perspective_images, dtype = np.float32)    

print("Done")

print("Stacking perspective images...")

X_train=np.vstack((X_train,train_perspective_images))
Y_train=np.vstack((Y_train,train_labels))

print("Done")

print("Finished!")
