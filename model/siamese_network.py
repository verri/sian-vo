# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

def build_siamese_model(inputShape=(64, 64, 1), size_kernel=(7, 7)):
	# specify the inputs for the feature extractor network
	inputs = Input(inputShape)
	chanDim = -1

	x = Conv2D(64, size_kernel, padding="same", activation="relu")(inputs)
	x = BatchNormalization(axis=chanDim)(x)
	x = Conv2D(64, size_kernel, padding="same", activation="relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = MaxPooling2D(pool_size=2)(x)
	outputs = Dropout(0.2)(x)

	# build the model
	model = Model(inputs, outputs)

	# return the model to the calling function
	return model