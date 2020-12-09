from xailib.models.bbox import AbstractBBox


class sklearn_classifier_wrapper(AbstractBBox):
    def __init__(self, classifier):
        self.bbox = classifier

    def predict(self, X):
        return self.bbox.predict(X)

    def predict_proba(self, X):
        return self.bbox.predict_proba(X)
