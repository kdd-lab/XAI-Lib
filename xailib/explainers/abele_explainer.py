from xailib.xailib_image import ImageExplainer, ImageExplanation
from xailib.models.bbox import AbstractBBox
from externals.ABELE.ilore.ilorem import ILOREM
from externals.ABELE.ilore.util import neuclidean


class ABELEImageExplanation(ImageExplanation):
    def __init__(self, abele_exp):
        super().__init__()
        self.exp = abele_exp

    def getFeaturesImportance(self, features=None, samples=400):
        return self.exp.get_image_rule(features=features, samples=samples)

    def getExemplars(self, num_prototypes):
        return self.exp.get_prototypes_respecting_rule(num_prototypes=num_prototypes)

    def getCounterExemplars(self):
        return self.exp.get_counterfactual_prototypes()

    def getRules(self):
        return self.exp.rstr()

    def getCounterfactualRules(self):
        return self.exp.cstr()

class ABELEImageExplainer(ImageExplainer):
    def __init__(self, bb: AbstractBBox):
        super().__init__()
        self.bb = bb

    def fit(self, config):
        self.exp = ILOREM(**config)
    
    def explain(self, img, num_samples=300, use_weights=True, metric=neuclidean):
        return ABELEImageExplanation(self.exp.explain_instance(img, num_samples=num_samples, use_weights=use_weights, metric=metric)) 

