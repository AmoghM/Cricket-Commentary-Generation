# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:39:51 2017

@author: HP
"""

import os,pickle
import h5py
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, LSTM, Reshape, RepeatVector
from keras import optimizers

# path to the model weights file.
weights_path = 'G:\Cricket_Annotation\\vgg16_weights.h5'

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'G:\Cricket_Annotation\\new_dataset'

nb_train_samples = 1272


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    #model.add(Reshape((512,8)))
    #model.add(LSTM(8,init='uniform'))
    #model.add(Dense(4,activation='relu'))

    #model.summary()
    #assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    print "nb_layers are",f.attrs['nb_layers']
    for k in range(f.attrs['nb_layers']):
        #print k
        if k >=36:#>= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=2,
            class_mode=None,
            shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    f=open('sample_data_train','w')
    pickle.dump(bottleneck_features_train,f)
    print np.shape(bottleneck_features_train),bottleneck_features_train
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    f.close()
    
def encoder_decoder():
    #sentences =[["a b c d e"],["1 2 3 4 5"],["6 7 8 9 10"],["f g h i j"],["11 12 13 14 15"]]
    #classes=[[0],[1],[2],[3],[1]]
    classes = [0]*40+[1]*40+[2]*40+[3]*39
    classes = np.array(classes)
    print np.shape(classes)
    classes=keras.utils.np_utils.to_categorical(classes,4)
    print np.shape(classes)
    dic={0:"boundary",1:"no run",2:"run",3:"wicket"}
    #print sentences
    #sentences = np.reshape(sentences,(1,5))
    #words= [[[0],[1],[2],[3],[4]],[[4],[1],[2],[3],[4]],[[0],[1],[2],[3],[4]],[[0],[1],[2],[3],[4]],[[4],[1],[2],[3],[4]]]
    #print "Shape of sentences is",np.shape(sentences)
    f=open('sample_data_train','r')
    image_features = pickle.load(f)
    input_image_features=[]
    #print np.shape(image_features)
    for i in range(0,1272,8):
        seq=image_features[i:i+8]
        #print np.shape(seq)
        input_image_features.append(seq)
    #print np.shape(input_image_features)
    l=np.array(input_image_features)
    #print l,'\n',sentences
    
    testFile = open('sample_datA_test','r')
    test = pickle.load(testFile)
    test_input=[]
    for i in range(0,160,8):
        seq=test[i:i+8]
        #print np.shape(seq)
        test_input.append(seq)
    test_balls = np.array(test_input)
    test_classes = [[0]]*5+[[1]]*5+[[2]]*5+[[3]]*5
    model = Sequential()
    model.add(LSTM(200,input_dim=4096,init='uniform',return_sequences=False))
    #model.add(Dropout((0.5)))
    #model.add(Dense(4,activation='softmax',init='uniform'))
    #model.summary()
    #sgd = optimizers.SGD(lr=1, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy',
    #          optimizer=sgd,metrics=['accuracy'])
    #print model.layers[0].get_weights()
    #model.fit(l,classes,batch_size=8,nb_epoch=1,shuffle=True)
    #print model.layers[0].get_weights()
    #print model.predict(test_balls)
    result = model.predict(l)
    verify = model.predict(test_balls)
    #print np.shape(result)
    #classify(result)
    
    classify = Sequential()
    classify.add(Dense(4,input_shape=(200,),activation='softmax',init='uniform'))
    classify.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    classify.fit(result,classes,batch_size=1,nb_epoch=10,shuffle=True)
    observed = classify.predict_classes(verify)
    
    hit=0.0
    miss=0.0
    #print "Five balls, each sentence having 5 words,each word of length 10",np.shape(result)
    for expected,actual in zip(test_classes,observed):
      print "Outcome is ", dic[actual] 
      if expected==actual:
           hit+=1
      else:
           miss+=1
    accuracy=hit/(hit+miss)
    print "Accuracy is: ", accuracy*100
#        print expected,actual
       
     
    
def classify(result):
    model = Sequential()
    model.add(Dense(4,input_shape=(200,),activation='softmax',init='uniform'))
    
def print_feature_vectors():
    f=open("sample_data_train","r")
    l=pickle.load(f)
    print np.shape(l)
    res = open("vectors","w")
    for i in l:
        res.write(str(i))
        res.write('\n')
#save_bottlebeck_features()
encoder_decoder()
#print_feature_vectors()
print "done"