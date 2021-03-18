#Copyright (C) 2020 BardzoFajnyZespol

# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

class Conv2DAutoencoder:

	@staticmethod
	def build(width, height, depth, filters=(32, 64), latentDim=16):
		# initialize the input shape to be "channels last" along with
		# the channels dimension itself
		# channels dimension itself
		inputShape = (height, width, depth)
		chanDim = -1

		# define the input to the encoder
		inputs = Input(shape=inputShape)
		i = inputs

		# loop over the number of filters
		for f in filters:
			# apply a CONV => RELU => BN operation
			i = Conv2D(f, (3, 3), strides=2, padding="same")(i)
			i = LeakyReLU(alpha=0.20)(i)
			i = BatchNormalization(axis=chanDim)(i)

		# flatten the network and then construct our latent vector
		volumeSize = K.int_shape(i)
		i = Flatten()(i)
		latent = Dense(latentDim)(i)

		# build the encoder model
		encoder = Model(inputs, latent, name="encoder")

		# start building the decoder model which will accept the
		# output of the encoder as its inputs
		latentInputs = Input(shape=(latentDim,))
		i = Dense(np.prod(volumeSize[1:]))(latentInputs)
		i = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(i)

		# loop over our number of filters again, but this time in
		# reverse order
		for f in filters[::-1]:
			# apply a CONV_TRANSPOSE => RELU => BN operation
			i = Conv2DTranspose(f, (3, 3), strides=2,
			                    padding="same")(i)
			i = LeakyReLU(alpha=0.20)(i)
			i = BatchNormalization(axis=chanDim)(i)

		# apply a single CONV_TRANSPOSE layer used to recover the
		# original depth of the image
		i = Conv2DTranspose(depth, (3, 3), padding="same")(i)
		outputs = Activation("sigmoid")(i)

		# build the decoder model
		decoder = Model(latentInputs, outputs, name="decoder")

		# our autoencoder is the encoder + decoder
		autoencoder = Model(inputs, decoder(encoder(inputs)),
		                    name="autoencoder")
		#autoencoder.summary()
		#decoder.summary()
		# return a 3-tuple of the encoder, decoder, and autoencoder
		return (encoder, decoder, autoencoder)
