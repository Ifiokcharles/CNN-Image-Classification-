#import the keras lib
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#intiate the cnn
classifierCNN = Sequential()

#convolution layer
classifierCNN.add(Convolution2D(32,(3, 3), input_shape =(64, 64, 3), activation = 'relu'))# coloured images are converted to a 3d array

#pooling layer
classifierCNN.add(MaxPooling2D(pool_size = (2,2)))

#flatten layer
classifierCNN.add(Flatten())

# make a full connection
classifierCNN.add(Dense(units = 128, activation = 'relu'))
classifierCNN.add(Dense(units = 1, activation = 'sigmoid'))

#compiling the cnn
classifierCNN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#part 2 - fitiing the cnn to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set_seg = train_datagen.flow_from_directory('intel-image-classification/seg_train/seg_train/',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set_seg = test_datagen.flow_from_directory('intel-image-classification/seg_test/seg_test/',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifierCNN.fit_generator(training_set_seg,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set_seg,
                        validation_steps=2000)

#part 3 -- making new predictions
import numpy as np
from keras.preprocessing import image
test_imageCNN = image.load_img('intel-image-classification/seg_pred/seg_pred/3.jpg', target_size = (64, 64))
test_imageCNN = image.img_to_array(test_imageCNN) #this helps create the '3' dimensinoal array (64, 64, '3')
test_imageCNN = np.expand_dims(test_imageCNN, axis = 0)#add a new dimension bcos of batch
result = classifierCNN.predict(test_imageCNN)
training_set_seg.class_indices
