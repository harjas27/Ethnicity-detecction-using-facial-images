from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras import optimizers
import numpy as np


img_width =250
img_height = 250
test_dir = 'F:/datasetss/Test'   
train_dir = 'F:/datasetss/Train'
nb_test = 1199
nb_train = 4867
epochs = 20
batch_size = 15
input_shape = (img_width,img_height,3)

def pre_training():
    datagen = ImageDataGenerator(rescale = 1./255)
    model = VGG16(weights = 'imagenet',include_top = False)
    generator = datagen.flow_from_directory(
                train_dir,
                target_size = (img_width,img_height),
                batch_size = batch_size,
                class_mode = 'categorical',
                shuffle = False)
    features_train = model.predict_generator(generator,nb_train//batch_size)
    np.save(open('features_train_ml_vgg16.npy','wb'),features_train)
    
    generator = datagen.flow_from_directory(
                test_dir,
                target_size = (img_width,img_height),
                batch_size = batch_size,
                class_mode = 'categorical',
                shuffle = False)
    features_test = model.predict_generator(generator,nb_test//batch_size)
    np.save(open('features_test_ml_vgg16.npy','wb'),features_test)

#pre_training()

def transfer_learning():
    train_data = np.load(open('features_train_ml_vgg16.npy','rb'))
    
    train_labels=[]
    
    train_labels.append([0]*750)
    train_labels.append([1]*1031)
    train_labels.append([2]*2691)
    #train_labels.append([3]*395)
    
    train_labels=np.array(train_labels)
    train_labels=np.hstack(train_labels)
    
    # print(train_data.shape)
    # print(train_labels.shape)
    test_data = np.load(open('features_test_ml_vgg16.npy','rb'))
    
    test_labels=[]
    test_labels.append([0]*176)
    test_labels.append([1]*253)
    test_labels.append([2]*671)
    #test_labels.append([3]*99)
    
    test_labels=np.array(test_labels)
    test_labels=np.hstack(test_labels)
    
#    print(test_data.shape)
#    print(test_labels.shape)
    model = Sequential()
    #model.add(Flatten(input_shape = train_data.shape[1:]))
   # model.add(Flatten(input_shape = train_data.shape[1:]))
    model.add(Dense(200,activation='relu',input_shape=train_data.shape[1:])) 
    #model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    
    model.compile(loss = 'sparse_categorical_crossentropy',
                optimizer = 'nadam',
                metrics = ['accuracy'])
    
    model.fit(train_data,train_labels,
             epochs = epochs,
             batch_size = batch_size,
             validation_data = (test_data,test_labels))
 

    fname=("weights_ml_3.h5")
    model.save(fname, overwrite=True)

transfer_learning()
