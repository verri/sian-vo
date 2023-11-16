from model.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.models import load_model
import numpy as np
import csv

SIZE_IMAGE = 32

test_sets = [
    'Downtown1',
    'Downtown2',
    'Downtown3',
    'Forest1',
    'Forest2',
    'Forest3',
    'Mountains1',
    'Mountains2',
    'Mountains3',
]

for ENVIRONMENT in test_sets:

    MODEL_PATH = f"checkpoints_{SIZE_IMAGE}/model_checkpoint"
# MODEL_PATH = f"checkpoints_{SIZE_IMAGE}/position_pred_model"
# MODEL_PATH = "checkpoints_norte/model_checkpoint"
    TEST_HDF5 = f"dataset_test_{ENVIRONMENT}_images_0_simulated_{SIZE_IMAGE}x{SIZE_IMAGE}.hdf5"
# TEST_HDF5 = "dataset_test_Forest2_images_0_simulated_norte_64x64.hdf5"
    BATCH_SIZE = 1

    FILE_OUTPUT = f'eval_model_simulated_{ENVIRONMENT}_{SIZE_IMAGE}x{SIZE_IMAGE}.csv'
# FILE_OUTPUT = 'eval_model_simulated_Forest2_norte_64x64.csv'

# create CSV file and write header to it
    header = ['predito', 'esperado', 'erro']

    with open(FILE_OUTPUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# load the pre-trained network
    print("[INFO] loading model...")
    model = load_model(MODEL_PATH)

    print("[INFO] loading testing generator...")
    testGen = HDF5DatasetGenerator(TEST_HDF5, BATCH_SIZE)


    gen_test = testGen.generator()

    trajectory_length = 0
    accumulated_error = 0
    n_pairs = 0
    for i in range(testGen.numImages // BATCH_SIZE):

        image_pair, label = next(gen_test)

        predictions = model.predict(image_pair)
        label = np.reshape(label, (-1, 1))

        error = predictions - label
        accumulated_error = accumulated_error + np.sum(error)
        trajectory_length = trajectory_length + np.sum(label)

        n_pairs = n_pairs + image_pair[0].shape[0]

        text = [float(predictions[0]), float(label[0]), float(error[0])]
        with open(FILE_OUTPUT, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(text)

    print("[INFO] Accumulated error {} m in {} images of straight line flight, comprising a trajectory of {} m...".format(accumulated_error, n_pairs, trajectory_length))
    print("[INFO] Mean error of: {} /m...".format(accumulated_error/trajectory_length))

    testGen.close()
