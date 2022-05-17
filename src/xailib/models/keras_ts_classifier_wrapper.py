from xailib.models.bbox import AbstractBBox
import numpy as np


class keras_classifier_wrapper(AbstractBBox):
    def __init__(self, classifier):
        super().__init__()
        self.bbox = classifier

    def model(self):
        return self.bbox

    def predict(self, X):
        #here the input is 3 dimensions
        y = self.bbox.predict(X)
        #not sure about this condition, check it
        if len(y.shape) > 1 and (y.shape[1] != 1):
            y = np.argmax(y, axis=1)
        return y.ravel()

    def predict_proba(self, X):
        #keras does not return predict_proba. the probabilities are in the predict
        return self.bbox.predict(X)
