import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import cv2
import random
from keras.utils.io_utils import HDF5Matrix

import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D, Flatten, Dense, Dropout, Activation, Input
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16, VGG19, InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping 
import warnings
warnings.filterwarnings('ignore')

#models_filename = 'E:/project/food101-tensorflow/v8_vgg16_model_1.h5'
image_dir = 'E:/project/food101-tensorflow/train20' #change
image_size = (224, 224)
batch_size = 64
#epochs = 1

# 5gb of images won't fit in my memory. use datagenerator to go across all images.
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = False,
fill_mode = "nearest",
zoom_range = 0,
width_shift_range = 0,
height_shift_range=0,
rotation_range=0)

train_generator = train_datagen.flow_from_directory(
image_dir,
target_size = (image_size[0], image_size[1]),
batch_size = batch_size,
color_mode= "grayscale", #trying to save memory 
class_mode = "categorical")

Validation = train_datagen.flow_from_directory(image_dir,
    (image_size[0], image_size[1]),
    batch_size = batch_size,
    color_mode= "grayscale", #trying to save memory
    class_mode = 'categorical')

num_of_classes = len(train_generator.class_indices)


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution (224,224, 3) for color
classifier.add(Conv2D(32, (5, 5), input_shape = (224, 224, 1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2), strides= 2))


# Adding a second convolutional layer
classifier.add(Conv2D(64, (5, 5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides= 2))

# Adding a third convolutional layer
classifier.add(Conv2D(128, (5, 5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides= 2))

# Adding a third convolutional layer
classifier.add(Conv2D(256, (5, 5), activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides= 2))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = num_of_classes, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#steps_per_epoch, validation_steps = num_images_total / batch_size 

history = classifier.fit_generator(train_generator, steps_per_epoch = 312, epochs = 50, validation_data = Validation, validation_steps = 312)
classifier.summary()
# Save the model after training
classifier.save('E:/project/test_models/20class650epoch.h5') #change

#Plotting graphs
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Validation acc')
plt.plot(epochs, val_acc, 'b', label='Training acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Validation loss')
plt.plot(epochs, val_loss, 'b', label='Training loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

preds = classifier.evaluate_generator(train_generator, steps=50, workers=1, use_multiprocessing=False)
preds

# Testing
#change
food_classes = ["apple_pie", "baby_back_ribs", "breakfast_burrito","cheesecake", "chicken_wings","cup_cakes","donuts","edamame","eggs_benedict","falafel","fish_and_chips", "garlic_bread","hamburger","hot_dog","ice_cream","lasagna", "pizza", "samosa", "steak", "sushi"]

classes_folder = "E:/project/food101-tensorflow/train20/" #change

for i in range(10):
    class_choosen = random.choice(food_classes)
    class_path = classes_folder + class_choosen
    #print(image_path)
    image_list = []
    for image in os.listdir(class_path):
      image_list.append(image)
    
    print(class_choosen)
    image_test = class_path + "/" + random.choice(image_list)
    #print(image_test)
    
    # Load image
    img = load_img(image_test, target_size=(224, 224))
    arr_img = img_to_array(img)

    predicted = classifier.predict(np.asarray([arr_img]))
    print(predicted)
    predicted_answer_index = np.argmax(predicted[0])
    predicted_answer = food_classes[predicted_answer_index]

    plt.imshow(arr_img/255)
    plt.title("actual class : "+class_choosen +" predicted class : "+predicted_answer)
    plt.show()

'''
# routine for human evaluation - use the generator so we can see how well it can predict
for n in range(10):
    _ = train_generator.next()
    image, clazzifier = (_[0][0],_[1][0]) # take the first image from the batch
    index = np.argmax(clazzifier)
    answer = list(train_generator.class_indices.keys())[index]
    predicted = classifier.predict(np.asarray([image]))
    predicted_answer_index = np.argmax(predicted[0])
    predicted_answer = list(train_generator.class_indices.keys())[predicted_answer_index]

    plt.imshow(image)
    plt.show()

    print()
    print('correct answer is: ', answer)
    print('CNN thinks it''s:', predicted_answer)
'''