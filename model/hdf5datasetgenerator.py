import numpy as np
import math
import h5py

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize=None):
        self.db = h5py.File(dbPath, "r")
        self.numImages = self.db["labels1"].shape[0]

        if batchSize is not None:
            self.batchSize = batchSize
        else:
            self.batchSize = self.numImages

    def random_generator(self, passes=np.inf):
        epochs = 0

        while epochs < passes:

            ix = np.arange(self.numImages)
            np.random.shuffle(ix)

            for i in np.arange(0, self.numImages, self.batchSize):
                images1 = self.db["images1"][np.sort(ix[i: i+self.batchSize])]
                images2 = self.db["images2"][np.sort(ix[i: i+self.batchSize])]
                images3 = self.db["images3"][np.sort(ix[i: i+self.batchSize])]
                images4 = self.db["images4"][np.sort(ix[i: i+self.batchSize])]
                labels1 = self.db["labels1"][np.sort(ix[i: i+self.batchSize])]

                images1 = images1.astype("float")/255.0
                images2 = images2.astype("float")/255.0
                images3 = images3.astype("float")/300.0
                images4 = (images4.astype("float") + math.pi) / (2 * math.pi)

                yield ([images1, images2, images3, images4], [labels1.astype("float")])

            epochs += 1

    def generator(self, passes=np.inf):
        epochs = 0

        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                images1 = self.db["images1"][i: i+self.batchSize]
                images2 = self.db["images2"][i: i+self.batchSize]
                images3 = self.db["images3"][i: i+self.batchSize]
                images4 = self.db["images4"][i: i+self.batchSize]
                labels1 = self.db["labels1"][i: i+self.batchSize]

                images1 = images1.astype("float")/255.0
                images2 = images2.astype("float")/255.0
                images3 = images3.astype("float")/300.0
                images4 = (images4.astype("float") + math.pi) / (2 * math.pi)

                yield ([images1, images2, images3, images4], [labels1.astype("float")])

            epochs += 1

    def close(self):
        self.db.close()
