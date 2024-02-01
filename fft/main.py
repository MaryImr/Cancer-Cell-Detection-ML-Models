from skimage.feature import hog
from PIL import Image
import os, os.path
import numpy as np
from scipy.fft import fft2, fftshift, fft
import cv2
import cmath

MIN_DESCRIPTOR = 18

# usually amplitude spectrum most useful with svm
# can also try power or phase spectrum

def findDescriptor(img):
    """ findDescriptor(img) finds and returns the
    Fourier-Descriptor of the image contour"""
    contour = []
    contour, hierarchy = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
        contour)
    contour_array = contour[0][:, 0, :]
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)
    return fourier_result

def truncate_descriptor(descriptors, degree):
    """this function truncates an unshifted fourier descriptor array
    and returns one also unshifted"""
    descriptors = np.fft.fftshift(descriptors)
    center_index = len(descriptors) / 2
    descriptors = descriptors[int(center_index - degree / 2):int(center_index + degree / 2)]
    descriptors = np.fft.ifftshift(descriptors)
    return descriptors

def addNoise(descriptors):
    """this function adds gaussian noise to descriptors
    descriptors should be a [N,2] numpy array"""
    scale = descriptors.max() / 10
    noise = np.random.normal(0, scale, descriptors.shape[0])
    noise = noise + 1j * noise
    descriptors += noise

def convertPolar(descriptors):
    descriptorsPolarR = []
    descriptorsPolarPhi = []
    for des in descriptors:
        r, phi = cmath.polar(des)
        descriptorsPolarR.append(r)
        descriptorsPolarPhi.append(phi)

    return descriptorsPolarR, descriptorsPolarPhi


imgs_precan = []
path = "/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/Precancer_generated"
valid_images = [".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    # imgs_precan.append(Image.open(os.path.join(path,f)))
    imgs_precan.append(cv2.imread(os.path.join(path, f), 0))

features_matrix = np.empty((180, 65341), dtype=float)
labels_matrix = np.empty(180, dtype=int)
i = 0

for img in imgs_precan:
    """
    f = fftshift(fft2(img))
    ampl_spectrum = np.abs(f)
    features_matrix[i] = ampl_spectrum
    labels_matrix[i] = 1
    i += 1
"""
    descriptors = findDescriptor(img)
    descriptors = truncate_descriptor(
        descriptors,
        MIN_DESCRIPTOR)
    addNoise(descriptors)
    descriptorsR, descriptorsPhi = convertPolar(descriptors)
    fft1 = fft(img, None, 1, None)
    fft1 = fft1.flatten()
    features_matrix[i] = fft1
    labels_matrix[i] = 1
    i += 1

# reading normal images and finding vectors

imgs_normal = []
path = "/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/Normal_generated"
valid_images = [".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs_normal.append(cv2.imread(os.path.join(path, f), 0))

for img in imgs_normal:
    """
    f = fftshift(fft2(img))
    ampl_spectrum = np.abs(f)
    features_matrix[i] = ampl_spectrum
    labels_matrix[i] = 0
    i += 1
    """
    descriptors = findDescriptor(img)
    descriptors = truncate_descriptor(
        descriptors,
        MIN_DESCRIPTOR)
    addNoise(descriptors)
    descriptorsR, descriptorsPhi = convertPolar(descriptors)
    fft1 = fft(img, None, 1, None)
    fft1 = fft1.flatten()
    features_matrix[i] = fft1
    labels_matrix[i] = 0
    i += 1

np.save('fourier_descriptors_1d_vertical_train_features', features_matrix)
np.save('fourier_descriptors_1d_vertical_train_labels', labels_matrix)

# test set
# reading precancer images and finding vectors for test set

imgs_precan = []
path = "/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/Precancer_generated_test"
valid_images = [".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs_precan.append(cv2.imread(os.path.join(path, f), 0))

features_matrix = np.empty((17, 65341), dtype=float)
labels_matrix = np.empty(17, dtype=int)
i = 0

for img in imgs_precan:
    """
    f = fftshift(fft2(img))
    ampl_spectrum = np.abs(f)
    features_matrix[i] = ampl_spectrum
    labels_matrix[i] = 1
    i += 1
"""

    descriptors = findDescriptor(img)
    descriptors = truncate_descriptor(
        descriptors,
        MIN_DESCRIPTOR)
    addNoise(descriptors)
    descriptorsR, descriptorsPhi = convertPolar(descriptors)
    fft1 = fft(img, None, 1, None)
    fft1 = fft1.flatten()
    features_matrix[i] = fft1
    labels_matrix[i] = 1
    i += 1

# reading normal images and finding vectors for test set

imgs_normal = []
path = "/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/Normal_generated_test"
valid_images = [".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs_normal.append(cv2.imread(os.path.join(path, f), 0))

for img in imgs_normal:
    """
    f = fftshift(fft2(img))
    ampl_spectrum = np.abs(f)
    features_matrix[i] = ampl_spectrum
    labels_matrix[i] = 0
    i += 1
"""

    descriptors = findDescriptor(img)
    descriptors = truncate_descriptor(
        descriptors,
        MIN_DESCRIPTOR)
    addNoise(descriptors)
    descriptorsR, descriptorsPhi = convertPolar(descriptors)
    fft1 = fft(img, None, 1, None)
    fft1 = fft1.flatten()
    features_matrix[i] = fft1
    labels_matrix[i] = 0
    i += 1

np.save('fourier_descriptors_1d_vertical_test_features', features_matrix)
np.save('fourier_descriptors_1d_vertical_test_labels', labels_matrix)
