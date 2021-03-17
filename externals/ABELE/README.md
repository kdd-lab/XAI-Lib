# ABELE
**A**dversarial **B**lack box **E**xplainer generating **L**atent **E**xemplars

**ABELE** is a local, model-agnostic explanation method able to overcome the existing limitations of the local approaches by exploiting the latent feature space, learned through an adversarial autoencoder, for the neighborhood generation process. Given an image classified by a given black box model, ABELE provides an explanation for the reasons of the proposed classification. The explanation consists of two parts: *(i)* a set of *exemplars* and *counter-exemplars* images illustrating, respectively, instances classified with the same label and with a different label than the instance to explain, which may be visually analyzed to understand the reasons for the classification, and *(ii)* a *saliency map* highlighting the areas of the image to explain that contribute to its classification, and areas of the image that push it towards another label.

*ABELE is implemented by the ilore module (image local rule based explainer).*

Requirements:
- python > 3.6
- sklearn
- matplotlib
- pandas
- skimage
- deap
- tensorflow
- keras

Dataset Links:
- [mnist](http://yann.lecun.com/exdb/mnist/)
- [cifar](https://www.cs.toronto.edu/~kriz/cifar.html)
- [fashion](https://www.kaggle.com/zalando-research/)

Black Box Links:
- [RF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html})
- [CNN](https://keras.io/examples/cifar10_cnn/)
- [ResNet](https://keras.io/examples/cifar10_resnet/)
