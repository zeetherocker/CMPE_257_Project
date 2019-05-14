# In[]:
import cv2
import time
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from IPython.display import clear_output
from sklearn.model_selection import train_test_split

from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler


# In[]:
IMG_SIZE = 50

TRAIN_SET_PATH = 'D:/College_Stuff/2nd_Sem/257/Project/traffic.tar/traffic/train/'
TEST_SET_PATH = 'D:/College_Stuff/2nd_Sem/257/Project/traffic.tar/traffic-small/test/'

TRAIN_LABELS_PATH = 'D:/College_Stuff/2nd_Sem/257/Project/traffic.tar/traffic/train.labels'
TEST_LABELS_PATH = 'D:/College_Stuff/2nd_Sem/257/Project/traffic.tar/traffic-small/test.labels'

trainLabels = open(TRAIN_LABELS_PATH, 'r').read().splitlines()
# trainLabels = trainLabels[:len(trainLabels)-1]
# testLabels = open('D:/College_Stuff/2nd_Sem/257/Assignments/Program 2/traffic/traffic-small/test.labels').read().split('\n')
# testLabels = testLabels[:len(testLabels)-1]

def updateProgress(curr, max):
    barLen = 20
    progress = curr/max

    if progress < 0: progress = 0
    elif progress > 1: progress = 1

    block = int(round(barLen * progress))
    clear_output()
    tenPercent = max * 0.01
    percent = progress * 100
    if (curr % tenPercent == 0):
        message = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (barLen - block), percent)
        print(message)

def doPreProcessing(images, path):
    data = []
    sizeOfData = len(images)
    for i, image in enumerate(images):
        img = cv2.imread(path + image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation= cv2.INTER_LINEAR)
        imgData, _ = hog(image, orientations=8, pixels_per_cell=(10, 10),cells_per_block=(2, 2),visualize=True,block_norm='L2')
        data.append([val[0] for val in imgData.tolist()])
        updateProgress(i, sizeOfData)
    return np.array(data)

trainImages = os.listdir(TRAIN_SET_PATH)
testImages = os.listdir(TEST_SET_PATH)


#%%
trainDataSet = doPreProcessing(trainImages, TRAIN_SET_PATH)

#%%
testDataSet = doPreProcessing(testImages, TEST_SET_PATH)

#%%
print(trainDataSet.shape)
print(testDataSet.shape)

#%%
scaledMatrix = StandardScaler().fit_transform(trainDataSet)

#%%

def getDimenReducer(type = 'KBest', **kwargs):
    randomState = 2
    options = {
        "PCA" : PCA(.95),
        "KBest": SelectKBest(chi2, k=kwargs['components'] if 'components' in kwargs else 10),
        "SVD": TruncatedSVD(n_components= kwargs['components'] if 'components' in kwargs else 2, random_state=randomState),
    }
    return options[type]


def getClassifier(type = 'EXT', **kwargs):
    randomState = 2
    allTypes = ['SVC', 'KNN', 'RF', 'EXT', 'MLP', 'SGD', 'ADA']
    options = {
        'SVC': SVC(random_state=randomState)
        'LinearSVC': LinearSVC(C=100, class_weight= 'balanced',
                         random_state=randomState),
        'KNN': KNeighborsClassifier(n_neighbors= kwargs['neighbors'] if 'neighbors' in kwargs else 10, n_jobs=-1),
        'RF': RandomForestClassifier(n_estimators= kwargs['neighbors'] if 'neighbors' in kwargs else 10, n_jobs=-1),
        'EXT': ExtraTreesClassifier(n_estimators= kwargs['neighbors'] if 'neighbors' in kwargs else 10,
                                    random_state=randomState, n_jobs=-1, class_weight='balanced_subsample',
                                    min_samples_split= kwargs['minSampleSplit'] if 'minSampleSplit' in kwargs else 2),
        'MLP': MLPClassifier(hidden_layer_sizes=kwargs['neighbors'], random_state=randomState),
        'SGD': SGDClassifier(n_jobs=-1, random_state= randomState, max_iter=50, class_weight=None),
        'ADA': AdaBoostClassifier(n_estimators=kwargs['neighbors'], random_state=randomState,
                                 base_estimator=DecisionTreeClassifier(min_samples_split=4, class_weight='balanced', random_state=randomState)),
        # 'Vote': VotingClassifier(estimators=[getClassifier('KNN', **kwargs), getClassifier('RF', **kwargs), getClassifier('EXT', **kwargs)], voting='soft', n_jobs=-1)
    }
    return options[type]

#%%

X_train, X_test, Y_train, Y_test = train_test_split(trainDataSet, trainLabels, test_size = 0.2, random_state=2)

#%%

dimenReducer = getDimenReducer('PCA', components=50)
X_train = dimenReducer.fit_transform(X_train, Y_train)
X_test = dimenReducer.transform(X_test, Y_test)
# reducedTest = dimenReducer.transform(testDataSet)
# X_train, X_test, Y_train, Y_test = train_test_split(reducedTrain, trainLabels, test_size = 0.2, random_state=2)

#%%

predictions = []
allClassifiers = ['SVC', 'LinearSVC', ]
for c in allClassifiers:
    classifier = getClassifier(c, neighbors=150, class_weight='balanced', minSampleSplit=2)
    classifier.fit(X_train, Y_train)
    joblib.dump(classifier, 'HOG_Model_150' + c +'.sav')
    predictions.append(classifier.predict(X_test))

#%%
print(" Evaluating classifier on test data ...")
for i, classifier in enumerate(allClassifiers):
    print("F1 for %s = %f" % (classifier, f1_score(Y_test, predictions[i], average='macro')))

from collections import Counter
finalPredictions = []
for i, _ in enumerate(predictions[0]):
    curr = []
    for row in predictions:
        curr.append(row[i])
    curr = Counter(curr).most_common()[0][0]
    finalPredictions.append(curr)

print("F1 for %s = %f" % (classifier, f1_score(Y_test, finalPredictions, average='macro')))
print(classification_report(Y_test, predictions[0]))
