import cv2
import numpy as np
import sklearn
import csv

# This functions returns image name from the full path of the image file
def get_image_name(full_path):
	return full_path.split('\\')[-1]

# This functions returns full path by adding current file path and image file director
def create_full_path(image_name):
	return r"data/IMG/" + image_name

def read_log_file(log_file, correction=0.2):
	images = []

	# Read data.csv file that contains simulator driving data
	with open(log_file) as file:
		
		read_file = csv.reader(file)
		next(read_file) # First row in the CSV file is heading
		
		for row in read_file:

			image_center_path = create_full_path(get_image_name(row[0])) # Image name is in first column
			image_left_path = create_full_path(get_image_name(row[1]))
			image_right_path = create_full_path(get_image_name(row[2]))
			measurement = float(row[3])

			images.append([image_center_path, "center", measurement])
			images.append([image_center_path, "flip", measurement * -1])
			images.append([image_left_path, "left", measurement + correction])
			images.append([image_right_path, "right", measurement - correction])
						
		return images

# This function reads driving log and prepares it for consumption to the model
# log_data = list of arrays containing rows of driving_log
# correction = correction parameter for right and left image correction
# batch_size = batch size to use to pre process data
def data_generator(log_data, batch_size=32):
	
	num_log_data = len(log_data)
	
	while 1: # This should loop forever... as and when new data is available, it will fetch and append them

		for offset in range(0, num_log_data, batch_size):

			batch = log_data[offset:offset+batch_size]

			feature = [] #Images are data
			label = [] #Steering wheel data is label for image data

			for row in batch:

				image = cv2.imread(row[0])
				# YUV color space is advised to be used in NVidia paper
				# image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

				if (row[1] == "flip"):					
					feature.append(np.fliplr(image))
				else:
					feature.append(image)

				label.append(row[2])

			# Keras expect numpy arrays
			X_train = np.array(feature)
			y_train = np.array(label)

			yield sklearn.utils.shuffle(X_train, y_train)



# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np

# img = mpimg.imread("data/IMG/center_2016_12_01_13_30_48_287.jpg")
# imgplot = plt.imshow(img)
# plt.show()

# cropped = img[61:140, :]
# imgplot = plt.imshow(cropped)
# plt.show()

# img = mpimg.imread("data/IMG/left_2016_12_01_13_30_48_287.jpg")
# imgplot = plt.imshow(img)
# plt.show()

# cropped = img[61:140, :]
# imgplot = plt.imshow(cropped)
# plt.show()

# img = mpimg.imread("data/IMG/right_2016_12_01_13_30_48_287.jpg")
# imgplot = plt.imshow(img)
# plt.show()

# cropped = img[61:140, :]
# imgplot = plt.imshow(cropped)
# plt.show()
