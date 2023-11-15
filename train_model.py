import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from model.siamese_network import build_siamese_model
from model.trainingMonitor import TrainingMonitor
from model.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, KLDivergence
from tensorflow.keras.optimizers import Adam


import time
import os

import logging

logging.basicConfig()
logging.root.setLevel(level=logging.INFO)

SIZE_IMAGE = 32
SIZE_KERNEL = (7, 7)

NUMBER_REPEAT = 5

IMG_SHAPE = (SIZE_IMAGE, SIZE_IMAGE, 1)
chanDim = -1
BATCH_SIZE = 128
# BATCH_SIZE = 1
EPOCHS = 50
MODEL_PATH = f"checkpoints_{SIZE_IMAGE}/position_pred_model"
FIG_PATH = "plots/monitor_position_{}.png".format(os.getpid())
TRAIN_HDF5 = f"train_dataset_simulated_{SIZE_IMAGE}x{SIZE_IMAGE}.hdf5"
VALID_HDF5 = f"val_dataset_simulated_{SIZE_IMAGE}x{SIZE_IMAGE}.hdf5"

# configure the siamese network
logging.info("[INFO] build network...")
imgA = Input(shape=IMG_SHAPE)
imgB = Input(shape=IMG_SHAPE)
imgC = Input(shape=IMG_SHAPE)
imgD = Input(shape=IMG_SHAPE)
featureExtractor = build_siamese_model(IMG_SHAPE, SIZE_KERNEL)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

downscaling = AveragePooling2D(pool_size=2)
featsC = downscaling(imgC)
featsD = downscaling(imgD)

x = concatenate([featsA, featsB, featsC, featsD], axis=chanDim)
x = Conv2D(64, SIZE_KERNEL, padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = Conv2D(64, SIZE_KERNEL, padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = Conv2D(64, SIZE_KERNEL, padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = Conv2D(64, SIZE_KERNEL, padding="same", activation="relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.2)(x)

x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
output1 = Dense(1, activation="linear", name="output1")(x)

model = Model(inputs=[imgA, imgB, imgC, imgD], outputs=[output1])

trainGen = HDF5DatasetGenerator(TRAIN_HDF5, BATCH_SIZE)
valGen = HDF5DatasetGenerator(VALID_HDF5, BATCH_SIZE)

# best model checkpoint
ckp_path = f"checkpoints_{SIZE_IMAGE}/model_checkpoint"
mcp = ModelCheckpoint(filepath=ckp_path,
					save_weights_only=False,
					monitor="val_loss", # => val_mean_absolute_percentage_error
                    # monitor="val_mean_absolute_error",
					save_best_only=True,
					mode="auto",
					save_freq="epoch",
					verbose=1)

callbacks=[mcp, TrainingMonitor(FIG_PATH)]

logging.info("[INFO] compiling model...")

mse = MeanSquaredError()
model.compile(loss= {'output1':mse} , optimizer="adam", metrics=["mean_squared_error"])

# mae = MeanAbsoluteError()
# model.compile(loss= {'output1':mae} , optimizer="adam", metrics=[['mean_absolute_error']])

#mape = MeanAbsolutePercentageError()
#model.compile(loss= {'output1':mape} , optimizer=Adam(learning_rate=0.0000001), metrics=['mean_absolute_percentage_error', 'mean_absolute_error'])

# model.compile(loss= {'output1':mape} , optimizer="adam", metrics=['mean_absolute_percentage_error', 'mean_absolute_error'])


model.summary()
# kld = KLDivergence()
# model.compile(loss= {'output1':kld} , optimizer="adam", metrics=['kl_divergence'])

# model.compile(loss= {'output1':mae, 'output2':mae} , optimizer="adam", metrics=[['mean_absolute_error'], ['mean_absolute_error']])
# model.compile(loss= {'output1':mse} , optimizer=Adam(learning_rate=0.0001), metrics=["mean_squared_error"])

logging.info("[INFO] training model...")
gen_training = trainGen.random_generator()
gen_validation = valGen.generator()

model.fit(
	gen_training,
	steps_per_epoch = trainGen.numImages // BATCH_SIZE,
	validation_data= gen_validation,
	validation_steps = valGen.numImages // BATCH_SIZE,
	epochs = EPOCHS,
	max_queue_size = 10,
	callbacks = callbacks,
	verbose = 1)



logging.info("[INFO] serializing model")
model.save(MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()
