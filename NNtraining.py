# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 00:37:03 2019

@author: davim, matheussfarias
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout

from python_speech_features import mfcc

PATH1 = 'clap/training/'
PATH2 = 'not_clap/training/'
PATH3 = 'clap/test/'
PATH4 = 'not_clap/test/'

#Getting training files

dt=[]
lb=[]

for file in os.listdir(PATH1):
    f = open(PATH1+file,'rt')
    data = []
    v1=[]
    for i in f:
        v1.append(int(i))
    data = np.array(v1)
    data = data-np.mean(data)
    data = data/np.max(data)
    
    mfc = mfcc(data).flatten().tolist()
    
    dt.append(mfc)
    #dt.append(abs(data[500:1500]))
    #dt.append(data[500:1000])
    lb.append(1)

for file in os.listdir(PATH2):
    f = open(PATH2+file,'rt')
    data = []
    v1=[]
    for i in f:
        v1.append(int(i))
    data = np.array(v1)
    data = data-np.mean(data)
    data = data/np.max(data)
    
    mfc = mfcc(data).flatten().tolist()
    
    dt.append(mfc)
    #dt.append(abs(data[500:1500]))
    #dt.append(data[500:1000])
    lb.append(0)
    
#Getting test files
    
dt_test=[]
lb_test=[]

for file in os.listdir(PATH3):
    f = open(PATH3+file,'rt')
    data = []
    v1=[]
    for i in f:
        v1.append(int(i))
    data = np.array(v1)
    data = data-np.mean(data)
    data = data/np.max(data)
    
    mfc = mfcc(data).flatten().tolist()
    
    dt_test.append(mfc)
    
    #dt_test.append(abs(data[500:1500]))
    #dt_test.append(data[500:1000])
    lb_test.append(1)

for file in os.listdir(PATH4):
    f = open(PATH4+file,'rt')
    data = []
    v1=[]
    for i in f:
        v1.append(int(i))
    data = np.array(v1)
    data = data-np.mean(data)
    data = data/np.max(data)
    
    mfc = mfcc(data).flatten().tolist()
    
    dt_test.append(mfc)
    
    #dt_test.append(abs(data[500:1500]))
    #dt_test.append(data[500:1000])
    lb_test.append(0)


#to numpy array
x_train = np.row_stack(dt)
y_train = np.row_stack(lb)
x_test = np.row_stack(dt_test)
y_test = np.row_stack(lb_test)

#Neural Network Architecture
model = None
model = Sequential()
model.add(Dense(200, input_dim=637, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#Training network
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history=model.fit(x_train, y_train,
          epochs=100,
          batch_size=16,validation_data=(x_test, y_test))
#score = model.evaluate(x_test, y_test, batch_size=128)
#plot_model(model, to_file='model.png',rankdir='LR',show_shapes=True)

#history = model.fit(x_train, y_train,
#          epochs=25,
#          batch_size=32,validation_data=(x_test, y_test))

#model.fit(validation_data=(X_test, Y_test))

_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy Teste: %.2f' % (accuracy*100))

_, accuracy = model.evaluate(x_train, y_train)
print('Accuracy Treinamento: %.2f' % (accuracy*100))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

conf_matrix=np.array([[0,0],[0,0]])
predict = model.predict(x_train)
for i in range(len(x_train)):
    if predict[i]>0.5:
        if y_train[i]==1:
            conf_matrix[1,1]+=1 
        else: 
            conf_matrix[0,1]+=1
    else:
        if y_train[i]==0:
            conf_matrix[0,0]+=1 
        else:
            conf_matrix[1,0]+=1
    
print(conf_matrix)


conf_matrix=np.array([[0,0],[0,0]])
predict = model.predict(x_test)
for i in range(len(x_test)):
    if predict[i] > 0.5:
        if y_test[i]==1:
            conf_matrix[1,1]+=1 
        else: 
            conf_matrix[0,1]+=1
            print(i)
    else:
        if y_test[i]==0:
            conf_matrix[0,0]+=1 
        else:
            conf_matrix[1,0]+=1
    
print(conf_matrix)