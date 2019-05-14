# In[]:

import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam


# In[]:

# dimensions of our images.
IMG_WIDTH, IMG_HEIGHT = 100, 100

TRAIN_SET_PATH = 'D:/College_Stuff/2nd_Sem/257/Project/traffic.tar/traffic/train/'
TEST_SET_PATH = 'D:/College_Stuff/2nd_Sem/257/Project/traffic.tar/traffic-small/test/'

TRAIN_LABELS_PATH = 'D:/College_Stuff/2nd_Sem/257/Project/traffic.tar/traffic/train.labels'
TEST_LABELS_PATH = 'D:/College_Stuff/2nd_Sem/257/Project/traffic.tar/traffic-small/test.labels'

trainSetFileNames = os.listdir(TRAIN_SET_PATH)
testSetFileNames = os.listdir(TEST_SET_PATH)

trainLabelsInput = list(map(int, open(TRAIN_LABELS_PATH, 'r').read().splitlines()))
testLabelsInput = list(map(int, open(TEST_LABELS_PATH, 'r').read().splitlines()))

NUM_CLASSES = len(set(trainLabelsInput))
NUM_CHANNELS = 1

# trainDF = pd.DataFrame({'filename':trainSetFileNames, 'class': trainLabelsInput})
# testDF = pd.DataFrame({'filename':testSetFileNames, 'class': testLabelsInput})

# In[]:

def dataPreProcess(path, fileNames):
    imgList = []
    for file in fileNames:
        img = cv2.imread(path + file, cv2.IMREAD_GRAYSCALE)
        # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgResize = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        imgList.append(imgResize)
    imgList = np.array(imgList).astype('float32')
    imgList /= 255
    return imgList

# In[]:
trainDataMatrix = dataPreProcess(TRAIN_SET_PATH, trainSetFileNames)
testDataMatrix = dataPreProcess(TEST_SET_PATH, testSetFileNames)

# In[]:
trainLabelsArr = np.array([val - 1 for val in trainLabelsInput])
testLabelsArr = np.array([val - 1 for val in testLabelsInput])

weights = compute_class_weight(class_weight='balanced', classes=np.unique(trainLabelsArr), y=trainLabelsArr)

trainLabels = np_utils.to_categorical(trainLabelsArr, NUM_CLASSES)
testLabels = np_utils.to_categorical(testLabelsArr, NUM_CLASSES)

# In[]:

epochs = 20
batch_size = 16

# In[]:

if NUM_CHANNELS == 1:
	if K.image_dim_ordering()=='th':
		trainDataSet = np.expand_dims(trainDataMatrix, axis=1)
		testDataSet = np.expand_dims(testDataMatrix, axis=1)
	else:
		trainDataSet= np.expand_dims(trainDataMatrix, axis=4)
		testDataSet= np.expand_dims(testDataMatrix, axis=4)
else:
	if K.image_dim_ordering()=='th':
		trainDataSet = np.rollaxis(trainDataMatrix, 3, 1)
		testDataSet = np.rollaxis(testDataMatrix, 3, 1)

trainDataSet[0].shape

# In[]:
# Defining the model
input_shape = trainDataSet[0].shape

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))

# In[]:

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])


# In[]:
from keras import callbacks

filename='model_train_new.csv'

csv_log = callbacks.CSVLogger(filename, separator=',', append=False)

early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='min')

filepath = "Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacksList = [csv_log, early_stopping, checkpoint]


# In[]:

hist = model.fit(trainDataSet,
                 trainLabels,
                 batch_size= batch_size,
                 epochs=epochs,
                 verbose= 1,
                 validation_data=(testDataSet, testLabels),
                 class_weight=weights,
                 callbacks= callbacksList)

# In[]:

model = load_model('Best-weights-my_model-001-1.0070-0.6703.hdf5')

# In[]:

score = model.evaluate(testDataSet, testLabels, batch_size=16)
score

# In[]:

predictions = model.predict_classes(testDataSet)

# In[]:

from sklearn.metrics import classification_report, f1_score

print(f1_score(testLabelsArr, predictions, average='macro'))
