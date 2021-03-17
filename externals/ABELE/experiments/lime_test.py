import numpy as np
import matplotlib.pyplot as plt


from skimage.color import label2rgb

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from experiments.exputil import get_dataset
from experiments.exputil import get_black_box


def main():

    dataset = 'mnist'
    black_box = 'RF'

    path = './'
    path_models = path + 'models/'

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)

    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)
    bb_predict, bb_predict_proba = get_black_box(black_box, black_box_filename, use_rgb)

    lime_explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

    i2e = 1
    img = X_test[i2e]

    exp = lime_explainer.explain_instance(img, bb_predict_proba, top_labels=1, hide_color=0, num_samples=1000,
                                          segmentation_fn=segmenter)
    print(exp.local_exp)
    print(exp.local_pred)

    # print(lime_explainer.Zlr)
    # print(lime_explainer.Zl)

    label = bb_predict(np.array([X_test[i2e]]))[0]
    print(label)

    # print(lime_explainer.Zl[:, label][0])
    # print(lime_explainer.lr.predict(lime_explainer.Zlr)[0])

    bb_probs = lime_explainer.Zl[:, label]
    lr_probs = lime_explainer.lr.predict(lime_explainer.Zlr)

    print(1 - np.sum(np.abs(np.round(bb_probs) - np.round(lr_probs))) / len(bb_probs))

    img2show, mask = exp.get_image_and_mask(Y_test[i2e], positive_only=False, num_features=5, hide_rest=False,
                                            min_weight=0.01)
    plt.imshow(label2rgb(mask, img2show, bg_label=0), interpolation='nearest')
    plt.show()

    img2show, mask = exp.get_image_and_mask(Y_test[i2e], positive_only=True, num_features=5, hide_rest=True,
                                            min_weight=0.01)
    plt.imshow(img2show.astype(np.int), cmap=None if use_rgb else 'gray')
    plt.show()


if __name__ == "__main__":
    main()
