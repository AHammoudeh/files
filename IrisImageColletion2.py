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
    testing_file_1 = sorted(files)[7*i][:-7]+'1_'+str(randint(1,3))+'.bmp'
    files_testing.append(testing_file_1)
    files_training.remove(testing_file_1)
    testing_file_2 = sorted(files)[7*i][:-7]+'2_'+str(randint(1,4))+'.bmp'
    files_testing.append(testing_file_2)
    files_training.remove(testing_file_2)


r = 150.0 / 320.0
dim = (150, int(280 * r))


data_training = []
label_training = []
for f1 in files_training:
    img = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    data_training.append(resized)
    #data_training.append(resized[52:-52,52:-52])
    label_training = np.empty(shape=[0,1])
    cv2.imwrite("C:\\Users\\AH\\Desktop\\Iris\\train\\"+f1[-11:],resized)

    
data_testing = []
label_testing = []
for f1 in files_testing:
    img = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    data_testing.append(resized)#[52:-52,52:-52])
    label_testing.append(f1[-11:-8])
    cv2.imwrite("C:\\Users\\AH\\Desktop\\Iris\\test\\"+f1[-11:],resized)


del data_path, img, img_dir, f1, r, dim,i,testing_file_1,testing_file_2,resized
del files,files_testing,files_training
#%%

def data_Shuffle(data, Label):
    perm = list(range(np.shape(data)[0]))
    np.random.shuffle(perm)
    Label2 = [Label[index] for index in perm]
    data2 = [data[index] for index in perm]
    del perm
    return (np.asarray(data2)/255,  np.asarray(Label2).astype('int32')-1)

trainx, trainy = data_Shuffle(data_training, label_training)
trainx2=trainx.reshape([-1, 87, 100, 1])
from keras.utils import to_categorical
trainy2 = to_categorical(trainy, num_classes=np.max(trainy)+1 )

#del trainx,trainy

testx, testy = data_Shuffle(data_testing, label_testing)
testx=testx.reshape([-1, 87, 100, 1])
testy = to_categorical(testy, num_classes=np.max(testy)+1 )


#%%
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

datagen.fit(trainx2)

x_batches = trainx2
y_batches = trainy2

# fits the model on batches with real-time data augmentation:
#model1.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
#                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

epochs = 6
for e in range(epochs):
	print('Epoch', e)
	batches = 0
	per_batch = 10
	for x_batch, y_batch in datagen.flow(trainx2, testy, batch_size=per_batch):
		x_batches = np.concatenate((x_batches, x_batch), axis = 0)
		y_batches = np.concatenate((y_batches, y_batch), axis = 0)
		batches += 1
		if batches >= len(trainx2) / per_batch:
			# we need to break the loop by hand because
			# the generator loops indefinitely
			break
        
#%%
# PIL import Image
z=0;
for f1 in x_batches:
    #img = Image.fromarray(255*f1[:,:,0], 'L')
    im = np.array(f1[:,:,0]* 255, dtype = np.uint8)
    #img.save('C:\\Users\\AH\\Desktop\\Iris\\test\\'+str(z)+'.bmp')
    z=z+1
    cv2.imwrite("C:\\Users\\AH\\Desktop\\Iris\\test\\"+str(z)+'.bmp',im)#f1[:,:,:])



z=0;
for f1 in testx:
    #img = Image.fromarray(255*f1[:,:,0], 'L')
    im = np.array(f1[:,:,0]*255, dtype = np.uint8)
    #img.save('C:\\Users\\AH\\Desktop\\Iris\\test\\'+str(z)+'.bmp')   
    cv2.imwrite("C:\\Users\\AH\\Desktop\\Iris\\test\\"+ str(testy[z]+1)+"__"+str(z)+'.bmp',im)
    z=z+1
    #f1[:,:,:])



#%% CNN model building

#import keras
from keras.models import Sequential
from keras.layers import BatchNormalization,Activation, Dense , Flatten , Dropout, Conv2D , MaxPooling2D, AveragePooling2D

input_size = trainx2.shape[1:]

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(5,5), strides=(1,1),
                     padding='same', use_bias=False, #border_mode="same",
                    input_shape=input_size))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(32 , kernel_size = (3,3),padding='same', use_bias=False))#, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.05))

model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(32 , kernel_size = (3,3),padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(AveragePooling2D(pool_size=(3,3)))

model.add(Conv2D(32 , kernel_size = (3,3),padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(AveragePooling2D(pool_size=(3,3)))

model.add(Flatten())
model.add(Dense(80 , name="FC-layer"))
model.add(BatchNormalization(name="FC-layerN"))
model.add(Activation("relu"))
#model.add(Dense(64 , activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(108 , activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])

#%%
#model.summary()
model.trainable_weights

#%% CNN model training
#model_info = model.fit(trainx2, trainy2,validation_split=0.1,batch_size=5,epochs=20,verbose=1)#, callbacks=[history])
score_hist=[]
valid_hist= []
for epoch in range (60):
    model_info = model.fit(x_batches, y_batches,validation_split=0.20,batch_size= 5*(int(epoch/10)+2),epochs=1,verbose=0)#, callbacks=[history])
    #print('epoch no. ',epoch)
    score = model.evaluate(testx, testy, verbose=0)
    print('epoch no. ',epoch, 'valid.acc:',model_info.history['acc'][0],'   Test accuracy:', score[1])
    score_hist.append(score[1])
    valid_hist.append(model_info.history['acc'][0])
    if model_info.history['acc'][0]*100 >= 93 :
        break;#%%

score = model.evaluate(testx, testy, verbose=0)
print('Test accuracy:', score[1])
print('Test loss:', score[0])
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
conf_mat = confusion_matrix(testy, y_pred_reg_numeric)
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

#%%
model.save('model_file.h5')
from keras.models import load_model
my_model = load_model('my_model.h5')

# training ...
temp_weights = [layer.get_weights() for layer in model.layers]

for i in range(len(temp_weights)):
    model.layers[i].set_weights(temp_weights[i])
    
    
#%% FC-layer output
    
from keras.models import Model  
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('FC-layer').output)
intermediate_output_train = intermediate_layer_model.predict(trainx2)

intermediate_output_test = intermediate_layer_model.predict(testx)

#input_size2 = intermediate_output.shape[1:]


#%% preparing training example 50% same pairs, 50% different pairs
import itertools, random

def Siamese_Network(intermediate_output,label):
    values, counts = np.unique(label, return_counts=True)
    N = np.shape(label)[0]#data.shape[0]
    Siamese_same=[]
    Siamese_different=[]
   
    for i in values:
        foo_indexes = [n for n,x in enumerate(label) if x==i]
        combination=list(itertools.combinations(foo_indexes, 2))
    
        for ind in combination:
            example=np.append(intermediate_output[ind[0]],intermediate_output[ind[1]])
            Siamese_same.append(example)
    
        a = [1]
        while a!=[]:
            m=np.random.choice(range(N), np.shape(combination)[0], replace=False)
            a= [x for x in m if x in foo_indexes]
        for ind in m:
            example=np.append(intermediate_output[random.choice(foo_indexes)],intermediate_output[ind])
            Siamese_different.append(example)
                
    Siamese_data =Siamese_same+Siamese_different
    Siamese_size = np.shape(Siamese_data)[0]
    Siamese_Label= np.zeros(Siamese_size)
    Siamese_Label[0:int(Siamese_size/2)]=1
    
    del Siamese_same,Siamese_different,foo_indexes,combination,example,values,counts,m,a
    return (Siamese_data,Siamese_Label)

Siamese_training,Siamese_Label_training =Siamese_Network(intermediate_output_train,label_training)

Siamese_testing,Siamese_Label_testing =Siamese_Network(intermediate_output_test,label_testing)


"""values, counts = np.unique(label_training, return_counts=True)

N = trainx2.shape[0]
Siamese_same=[]
Siamese_different=[]

for i in values:
    foo_indexes = [n for n,x in enumerate(label_training) if x==i]
    combination=list(itertools.combinations(foo_indexes, 2))
    
    for ind in combination:
        training_example=np.append(intermediate_output_train[ind[0]],intermediate_output_train[ind[1]])
        Siamese_same.append(training_example)
    
    a = [1]
    while a!=[]:
        m=np.random.choice(range(N), np.shape(combination)[0], replace=False)
        a= [x for x in m if x in foo_indexes]
    for ind in m:
        training_example=np.append(intermediate_output_train[random.choice(foo_indexes)],intermediate_output_train[ind])
        Siamese_different.append(training_example)


Siamese_training =Siamese_same+Siamese_different

Siamese_size = np.shape(Siamese_training)[0]
Siamese_Label= np.zeros(Siamese_size)
Siamese_Label[0:int(Siamese_size/2)]=1"""

#%%
#from random import shuffle

#perm=shuffle(np.random.permutation(Siamese_size))
def Siamese_Shuffle(Siamese_data, Siamese_Label):
    perm = list(range(np.shape(Siamese_data)[0]))
    np.random.shuffle(perm)
    Siamese_Label = [Siamese_Label[index] for index in perm]
    Siamese_data = [Siamese_data[index] for index in perm]
    del perm
    return (np.asarray(Siamese_data), np.asarray(Siamese_Label))
    
Siamese_training, Siamese_Label_training = Siamese_Shuffle(Siamese_training, Siamese_Label_training)
Siamese_testing,Siamese_Label_testing = Siamese_Shuffle(Siamese_testing, Siamese_Label_testing)

"""perm = list(range(Siamese_size))
np.random.shuffle(perm)
Siamese_Label = [Siamese_Label[index] for index in perm]
Siamese_training = [Siamese_training[index] for index in perm]"""





#%%contrastive_loss
from keras import backend as K
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean((1-y_true) * K.square(y_pred) + (y_true) * K.square(K.maximum(margin - y_pred, 0)))

#%%
Siamese_input = Siamese_training[0].shape[0]

Siamese = Sequential()
Siamese.add(Dense(128 , input_shape=(Siamese_input,)))
Siamese.add(BatchNormalization())
Siamese.add(Activation("relu"))
#Siamese.add(Dropout(0.4))
#Siamese.add(Dense(128))
#Siamese.add(BatchNormalization())
#Siamese.add(Activation("tanh"))
#Siamese.add(Dense(128 , activation = 'relu'))
Siamese.add(Dense(1, activation='sigmoid'))
Siamese.compile(loss= contrastive_loss,#'binary_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])

 #%%
 
for epoch in range (300):
    Siamese_info = Siamese.fit(Siamese_training, Siamese_Label_training,validation_split=0.1,batch_size=64,epochs=1,verbose=1)#, callbacks=[history])
    print('epoch no. ',epoch)
    if Siamese_info.history['acc'][0]*100 >= 97.5 :
        break;
        
score2 = Siamese.evaluate(Siamese_testing,Siamese_Label_testing, verbose=0)
print('Test accuracy:', score2[1])
