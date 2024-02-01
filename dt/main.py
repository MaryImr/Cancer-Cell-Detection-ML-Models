import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree

"""
#for amplitude
train_set = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/fft/amplitude_spectrum_train_features.npy')
train_lbs = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/fft/amplitude_spectrum_train_labels.npy')
test_set = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/fft/amplitude_spectrum_test_features.npy')
test_lbs = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/fft/amplitude_spectrum_test_labels.npy')
"""
"""
#for phase
train_set = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/fft/phase_spectrum_train_features.npy')
train_lbs = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/fft/phase_spectrum_train_labels.npy')
test_set = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/fft/phase_spectrum_test_features.npy')
test_lbs = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/fft/phase_spectrum_test_labels.npy')
"""
"""
#for hog
train_set = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/HistOGrad/hog_feature_matrix_train.npy')
train_lbs = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/HistOGrad/hog_label_matrix_train.npy')
test_set = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/HistOGrad/hog_feature_matrix_test.npy')
test_lbs = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/HistOGrad/hog_label_matrix_test.npy')
"""
# """
train_set = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/fft/fourier_descriptors_1d_horizontal_train_features.npy')
train_lbs = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/fft/fourier_descriptors_1d_horizontal_train_labels.npy')
test_set = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/fft/fourier_descriptors_1d_horizontal_test_features.npy')
test_lbs = np.load(
    '/Users/maryamimran/Documents/Undergraduate/4th Year/8th Semester/ESC 492/fft/fourier_descriptors_1d_horizontal_test_labels.npy')
# """

train_set = np.append(train_set, test_set, axis=0)
train_lbs = np.append(train_lbs, test_lbs, axis=0)

# num_samples, height, width = train_set.shape
# train_set_2d = train_set.reshape(num_samples, height * width)

X_train, X_test, y_train, y_test = train_test_split(train_set, train_lbs)


params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random']
}

dt = tree.DecisionTreeClassifier()
grid_search = GridSearchCV(dt, params, cv=10, verbose=10, scoring='accuracy')
grid_search.fit(X_train, y_train)
print('CV Train accuracy: {:.2f}'.format(grid_search.best_score_))
print("Test accuracy: {:.2f}".format(grid_search.score(X_test, y_test)))
print('Best parameters: {}'.format(grid_search.best_params_))
