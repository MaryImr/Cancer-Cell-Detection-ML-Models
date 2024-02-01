from skimage.feature import hog
from PIL import Image
import os, os.path
import numpy as np
import cv2

#train set
#reading precancer images and finding vectors

imgs_precan = []
path = "/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/Precancer_generated"
valid_images = [".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs_precan.append(Image.open(os.path.join(path,f)))

features_matrix = np.empty((180, 2178), dtype=float)
labels_matrix = np.empty(180, dtype=int)
i=0

for img in imgs_precan:
    fd = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
    features_matrix[i] = fd
    labels_matrix[i] = 1
    i += 1


#reading normal images and finding vectors

imgs_normal = []
path = "/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/Normal_generated"
valid_images = [".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs_normal.append(Image.open(os.path.join(path,f)))

for img in imgs_normal:
    fd = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
    features_matrix[i] = fd
    labels_matrix[i] = 0
    i += 1

"""
#filtered
np.save('hog_filtered_feature_matrix_train',features_matrix)
np.save('hog_filtered_label_matrix_train',labels_matrix)
"""
np.save('hog_feature_matrix_test',features_matrix)
np.save('hog_label_matrix_test',labels_matrix)


#test set
#reading precancer images and finding vectors for test set

imgs_precan = []
path = "/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/Precancer_generated_test"
valid_images = [".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs_precan.append(Image.open(os.path.join(path,f)))

features_matrix = np.empty((17, 2178), dtype=float)
labels_matrix = np.empty(17, dtype=int)
i=0

for img in imgs_precan:
    fd = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
    features_matrix[i] = fd
    labels_matrix[i] = 1
    i += 1


#reading normal images and finding vectors for test set

imgs_normal = []
path = "/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/Normal_generated_test"
valid_images = [".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs_normal.append(Image.open(os.path.join(path,f)))

for img in imgs_normal:

    fd = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
    features_matrix[i] = fd
    labels_matrix[i] = 0
    i += 1

"""
#filtered
np.save('hog_filtered_feature_matrix_test',features_matrix)
np.save('hog_filtered_label_matrix_test',labels_matrix)
"""
np.save('hog_feature_matrix_test',features_matrix)
np.save('hog_label_matrix_test',labels_matrix)