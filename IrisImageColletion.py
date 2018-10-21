# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 14:15:33 2018

@author: AH
"""
import numpy as np
import cv2
import os
import glob
img_dir = "C:\\Users\\AH\\Desktop\\Iris\\V1" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*bmp')
files = glob.glob(data_path)

from random import randint
# generate some integers
files_training = files.copy()
files_testing =[]
for i in range(108):
    testing_file_1 = files[7*i][:-7]+'1_'+str(randint(1,3))+'.bmp'
    files_testing.append(testing_file_1)
    files_training.remove(testing_file_1)
    testing_file_2 = files[7*i][:-7]+'2_'+str(randint(1,4))+'.bmp'
    files_testing.append(testing_file_2)
    files_training.remove(testing_file_2)




r = 100.0 / 320.0
dim = (100, int(280 * r))

data_training = []
label_training = []

for f1 in files_training:
    img = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
    #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #data_training.append(resized)
    data_training.append(img[65:-65,65:-65])
    label_training.append(f1[-11:-8])
    cv2.imwrite("C:\\Users\\AH\\Desktop\\Iris\\train\\"+f1[-11:],img[65:-65,65:-65])

    
data_testing = []
label_testing = []
for f1 in files_testing:
    img = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
    #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    data_testing.append(img[65:-65,65:-65])
    label_testing.append(f1[-11:-8])
    cv2.imwrite("C:\\Users\\AH\\Desktop\\Iris\\test\\"+f1[-11:],img[65:-65,65:-65])


#scaling
#r = 100.0 / 320.0
#dim = (100, int(280 * r))
#resized = [cv2.resize(x, dim, interpolation = cv2.INTER_AREA) for x in data]

# perform the actual resizing of the image and show it
#cv2.imshow("resized", resized[1])
#cv2.waitKey(0)

del data_path, img, img_dir, f1, r, dim,i,testing_file_1,testing_file_2#,resized
#%%
trainx = np.asarray(data_training)
trainx2 = trainx.astype('float32')/255

#labels zero-based numbering
trainy = np.asarray(label_training).astype('int32')-1
from keras.utils import to_categorical
trainy2 = to_categorical(trainy, num_classes=trainy[-1]+1 )

del trainx,trainy

#%%
#CNN model building
#trainy1=trainy2.copy()
trainx2=trainx2.reshape([-1, 150, 190, 1])
#trainx1=trainx2.copy()

#trainx2=trainx1[0:100]
#trainy2=trainy2[0:100]

#import keras
from keras.models import Sequential
from keras.layers import Dense , Flatten , Dropout, Conv2D , MaxPooling2D, AveragePooling2D

input_size = trainx2.shape[1:]

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),
                     activation='relu', padding='same', #border_mode="same",
                    input_shape=input_size))

model.add(Conv2D(32 , kernel_size = (5,5),padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Flatten())
model.add(Dense(64 , activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(108 , activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])

#model.summary()

#%% CNN model training
from keras.callbacks import Callback
class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()


model_info = model.fit(trainx2, trainy2,validation_split=0.1,batch_size=10,epochs=30,verbose=1)#, callbacks=[history])
#model_info = model.fit(trainx2,validation_split=0.2, trainy2,batch_size=16,epochs=5,verbose=1)
#model_info = model.fit(trainx_2D, trainy_1H,batch_size=256,epochs=15,verbose=0)
import matplotlib.pyplot as plt
plt.plot(range(1,11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


#%%
testx = np.asarray(data_testing)
testx = testx.astype('float32')/255

#labels zero-based numbering
testy1 = np.asarray(label_testing).astype('int32')-1
testy = to_categorical(testy1, num_classes=testy1[-1]+1 )



score = model.evaluate(testx, testy, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%
#model predictions

y_pred = model.predict_proba(testx).ravel()
y_pred_1H = y_pred.reshape([testx.shape[0],108])
#y_pred_2H = y_pred.reshape([108,testx.shape[0]])

y_pred_1H_reg = np.zeros(y_pred_1H.shape)
#maxindex = y_pred_1H.argmax(axis=1)
#y_pred_1H_reg[y_pred_1H>0.5]=1
for i,a in enumerate(y_pred_1H):
    y_pred_1H_reg[i][a.argmax(axis=0)]=1
#convert 1 hot encoding back to single numeric 
y_pred_reg_numeric =    y_pred_1H.argmax(axis=1)
y_pred_reg_numeric = [ np.argmax(t) for t in y_pred_1H_reg ]


#%%
#model evaluation
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

#confusion matrix
conf_mat = confusion_matrix(testy1, y_pred_reg_numeric)

# accuracy
print("accuracy: ",accuracy_score(testy, y_pred_1H_reg))
#precision
print("precision matrix: ",precision_score(testy, y_pred_1H_reg, average=None))
print("average precision: ",precision_score(testy, y_pred_1H_reg, average="micro"))
#recall
print("recall matrix: ",recall_score(testy, y_pred_1H_reg, average=None))  
print("average recall: ",recall_score(testy, y_pred_1H_reg, average="micro"))  
# f1
print("f1 matrix: ",f1_score(testy, y_pred_1H_reg, average=None))
print("average f1: ",f1_score(testy, y_pred_1H_reg, average="micro"))













# GRADED FUNCTION: triplet_loss
import tensorflow as tf


"""    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
       y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)"""
            
def triplet_loss(y_true, y_pred, alpha = 0.2):            
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2] 
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    
    return loss




model.save('model_file.h5')
from keras.models import load_model
my_model = load_model('my_model.h5')