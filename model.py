# Third project for Udacity's Self Driving Car Nanodegree - Term 1

from lib import create_full_path, get_image_name
from lib import data_generator, read_log_file

# ========== Work on CSV file, getting input data ==========

import csv
# Correction parameter for left and right image correction
correction = 0.3

log_file = "data/driving_log.csv" # Path to simulator driving log file
log_data = read_log_file(log_file, correction) # List to contain all driving log data, row by row

# ========== Work on preparing features (images) and labels (steering value) ready ==========

from sklearn.model_selection import train_test_split

batch_size = 32

# This function includes shuffling by default
train_samples, validation_samples = train_test_split(log_data, test_size=0.2)

# compile and train the model using the generator function
train_generator = data_generator(train_samples, batch_size)
validation_generator = data_generator(validation_samples, batch_size)


# ========== Work on neural network ========== 

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, Dropout, MaxPooling2D, Cropping2D
from keras import backend as K

# Tried out simple one layer network to make sure everything is alright - kind of sanity testing :-)
# Have included loss results from this model in README.md file
# model = Sequential()
# model.add(Lambda(lambda x: (x/255) - 0.5, input_shape=(160, 320, 3))) # Normalizing data
# model.add(Flatten())
# model.add(Dense(1))

num_classes = 1
epochs = 5
keep_prob = 0.7

# input image dimensions
img_rows, img_cols, img_channels = 160, 320, 3
input_shape = (img_rows, img_cols, img_channels)

# NVIDIA-like implementation
model = Sequential()
model.add(Lambda(lambda x: (x/255) - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((70, 25), (0, 0)))) # 70 rows pixels from the top, 25 from bottom, 15 from left & right 
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
# model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(keep_prob))
model.add(Dense(10))
model.add(Dense(num_classes))

# # LeNet-like implementation
# model = Sequential()
# model.add(Lambda(lambda x: (x/255) - 0.5, input_shape=input_shape))
# model.add(Cropping2D(cropping=((70, 25), (0, 0)))) # 70 rows pixels from the top, 25 from bottom, 0 from left & right 
# model.add(Conv2D(32, 3, 3, activation='relu'))
# model.add(Conv2D(64, 3, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes)) # softmax is not relevent as this is regression model not a classifier model

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

model.compile(optimizer='adam', loss='mse')

# model.fit(X_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_split=0.2, 
#           shuffle=True)

# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)

model.fit_generator(train_generator, 
					samples_per_epoch=len(train_samples), 
					validation_data=validation_generator,
            		nb_val_samples=len(validation_samples), 
            		nb_epoch=epochs)


# r is added in front of the string to take this as a raw string so "unicodescape" error is avoided
# model.save(r"C:\Users\parva\dev\SDCND\CarND-Behavioral-Cloning-P3\model.h5")
model.save("model.h5")

