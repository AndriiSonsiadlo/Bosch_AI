#Copyright (C) 2020 BardzoFajnyZespol

# import the necessary packages
from imutils import paths
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import random
import pickle
import os
import cv2
import matplotlib

matplotlib.use("Agg")


def create_unsupervised_dataset(data, labels, validLabel: int = 1,
                                anomalyLabel: int = 3, contamAnomaly: float = 0.01,
                                seed=42, contamValid: float = 1.0):
	# grab all indexes of the supplied class label that are *truly*
	# that particular label, then grab the indexes of the image
	# labels that will serve as our "anomalies"
	validIdxs = np.where(labels == validLabel)[0]
	anomalyIdxs = np.where(labels == anomalyLabel)[0]

	# randomly shuffle both sets of indexes
	random.shuffle(validIdxs)
	random.shuffle(anomalyIdxs)

	# compute the total number of anomaly data points to select
	i = int(len(validIdxs) * contamAnomaly)
	anomalyIdxs = anomalyIdxs[:i]
	print(anomalyIdxs)

	i = int(len(validIdxs) * contamValid)
	validIdxs = validIdxs[:i]
	print(validIdxs)

	# use NumPy array indexing to extract both the valid images and
	# "anomlay" images
	validImages = data[validIdxs]
	anomalyImages = data[anomalyIdxs]

	# stack the valid images and anomaly images together to form a
	# single data matrix and then shuffle the rows
	images = np.vstack([validImages, anomalyImages])
	np.random.seed(seed)
	np.random.shuffle(images)

	# return the set of images
	return images


def main(contamAnomaly: float = 1, contamValid: float = 1, datasetPath: str = "dataset", imageDatasetPath:str = "output/image_my.pickle"):
	# initialize the data and labels
	data = []
	labels = []
	# grab the image paths and randomly shuffle them
	print("[INFO] loading images...")
	imagePaths = sorted(list(paths.list_images(datasetPath)))

	IMAGE_DIMS = [96, 96, 3]

	# loop over the input images
	for imagePath in imagePaths:
		# load the image, pre-process it, and store it in the data list
		image = cv2.imread(imagePath, 0)
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		image = img_to_array(image)
		data.append(image)

		# extract the class label from the image path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)

	# scale the raw pixel intensities to the range [0, 1]
	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)

	print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))
	# binarize the labels
	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)

	# build our unsupervised dataset of images with a small amount of
	# contamination (i.e., anomalies) added into it
	print("[INFO] creating unsupervised dataset...")
	images = create_unsupervised_dataset(data, labels, validLabel=0, anomalyLabel=1,
	                                     contamAnomaly=contamAnomaly, contamValid=contamValid)

	print("[INFO] saving image data...")
	f = open(imageDatasetPath, "wb")
	f.write(pickle.dumps(images))
	f.close()


if __name__ == '__main__':

	part = "rotated_tip_2"                                                # change only this varible

	main(
		datasetPath=f"datasets/dataset_{part}",                          # write from...
		imageDatasetPath=f"output/{part}/images_{part}.pickle"           # save as ...
	)
