import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import time
from xailib.models.bbox import AbstractBBox
from xailib.xailib_image import ImageExplainer

class RiseXAIImageExplainer(ImageExplainer):
    
    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.model = bb
        self.masks = None
    
    def fit(self, N, s, p1):
        self.N = N
        self.s = s
        self.p1 = p1
        cell_size = np.ceil(np.array(self.model.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.model.input_size))

        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + self.model.input_size[0], y:y + self.model.input_size[1]]
        self.masks = self.masks.reshape(-1, *self.model.input_size, 1)

    def explain(self, inp, batch_size=100):
        preds = []
        # Make sure multiplication is being done for correct axes
        masked = inp * self.masks
        for i in range(0, self.N, batch_size):
            preds.append(self.model.predict(masked[i:min(i+batch_size, self.N)]))
        preds = np.concatenate(preds)
        sal = preds.T.dot(self.masks.reshape(self.N, -1)).reshape(-1, *self.model.input_size)
        del preds
        sal = sal / self.N / self.p1
        return sal
