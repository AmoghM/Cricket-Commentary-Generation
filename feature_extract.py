import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, LSTM, Reshape
import matplotlib
import matplotlib.pyplot as plt
import cv2
import theano

# path to the model weights file.
weights_path = 'G:\Cricket_Annotation\\vgg16_weights.h5'

# dimensions of our images.
img_width, img_height = 224, 224
batch_size = 1

train_data_dir = 'G:\Cricket_Annotation\data\\train'

nb_train_samples = 1


img = cv2.imread('4.jpg')
frame = cv2.resize(img,(img_width, img_height), interpolation=cv2.INTER_AREA)
input = np.array(frame)
print input.shape
ipt = np.rollaxis(input,2)
print ipt.shape

f=[]
f.append(ipt)
l = np.array(f)
print l.shape
   
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

model.add(Reshape((512,8)))
model.add(LSTM(8,init='uniform'))
model.add(Dense(4,activation='relu'))
#model.add(LSTM(8, batch_input_shape=(batch_size,1, 1), stateful=True, return_sequences=True))
#model.add(LSTM(32, batch_input_shape=(batch_size,1, 1), stateful=True))
model.summary()
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

'''generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=2,
        class_mode=None,
        shuffle=False)'''
#bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
bottleneck_features_train = model.predict(l, nb_train_samples)
print np.shape(bottleneck_features_train),bottleneck_features_train
np.save(open('bottleneck_features_train_visualization.npy', 'w'), bottleneck_features_train)

    
def plot_filters(layer, x, y):
    '''Plot the filters for net after the (convolutional) layer number layer.
    They are plotted in x by y format. So, for example if we have 20 filters 
    after layer 0, then we can call plot_filters(l_convol, 5, 4) to get a 5 by 4
    plot of all filters.
    '''
    print "Hey.. Neerav!!!!"
    filters = layer.W.get_value()
    fig = plt.figure()
    for j in range(len(filters)):
        ax = fig.add_subplot(y,x, j+1)
        ax.matshow(filters[j][0], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    plt.show()

#input_image = save_bottlebeck_features()

plot_filters(model.layers[1], 8, 8)


#visualizing intermediate layers
output_layer = model.layers[1].output
output_fn = theano.function([model.layers[0].input], output_layer)

#the input image
# 1st input image 
input_image = l
print input_image
print input_image.shape

#visualize the input image in grey scale and in RGB
#plt.imshow(input_image[0,0,:,:], cmap='gray')
#plt.show()
plt.imshow(input_image[0,0,:,:])
plt.show()

print "-----------------------------------------------------"

output_image = output_fn(input_image)
print output_image.shape

#Rearrange dimension so we can plot the result as RGB image
output_image = np.rollaxis(np.rollaxis(output_image,3,1), 3,1)
print output_image.shape

fig = plt.figure(figsize = (8,8))
for i in range(64):
    ax = fig.add_subplot(8, 8, i+1)
    ax.imshow(output_image[0,:,:,i])
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
    
plt.show()
