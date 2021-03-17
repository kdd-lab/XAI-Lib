import numpy as np
import matplotlib.pyplot as plt

from skimage.color import gray2rgb
from skimage import feature, transform

from ilore.ilorem import ILOREM
from ilore.util import neuclidean

from experiments.exputil import get_dataset
from experiments.exputil import get_black_box
from experiments.exputil import get_autoencoder


import warnings
warnings.filterwarnings('ignore')


def main():

    random_state = 0
    dataset = 'fashion'
    black_box = 'RF'

    ae_name = 'aae'

    path = './'
    path_models = path + 'models/'
    path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)

    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)
    bb_predict, bb_predict_proba = get_black_box(black_box, black_box_filename, use_rgb)
    ae = get_autoencoder(X_test, ae_name, dataset, path_aemodels)
    ae.load_model()

    class_name = 'class'
    class_values = ['%s' % i for i in range(len(np.unique(Y_test)))]

    i2e = 10
    img = X_test[i2e]

    explainer = ILOREM(bb_predict, class_name, class_values, neigh_type='rnd', use_prob=True, size=1000, ocr=0.1,
                       kernel_width=None, kernel=None, autoencoder=ae, use_rgb=use_rgb, valid_thr=0.5,
                       filter_crules=True, random_state=random_state, verbose=True, alpha1=0.5, alpha2=0.5,
                       metric=neuclidean, ngen=10, mutpb=0.2, cxpb=0.5, tournsize=3, halloffame_ratio=0.1,
                       bb_predict_proba=bb_predict_proba)

    exp = explainer.explain_instance(img, num_samples=1000, use_weights=True, metric=neuclidean)

    print('e = {\n\tr = %s\n\tc = %s    \n}' % (exp.rstr(), exp.cstr()))
    print(exp.bb_pred, exp.dt_pred, exp.fidelity)
    print(exp.limg)

    img2show, mask = exp.get_image_rule(features=None, samples=10)
    if use_rgb:
        plt.imshow(img2show, cmap='gray')
    else:
        plt.imshow(img2show)
    bbo = bb_predict(np.array([img2show]))[0]
    plt.title('image to explain - black box %s' % bbo)
    plt.show()

    # if use_rgb:
    #     plt.imshow(img2show, cmap='gray')
    # else:
    #     plt.imshow(img2show)

    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, img2show.shape[1], dx)
    yy = np.arange(0.0, img2show.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)

    # Compute edges (to overlay to heatmaps later)
    percentile = 100
    dilation = 3.0
    alpha = 0.8
    xi_greyscale = img2show if len(img2show.shape) == 2 else np.mean(img2show, axis=-1)
    in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
    edges = feature.canny(in_image_upscaled).astype(float)
    edges[edges < 0.5] = np.nan
    edges[:5, :] = np.nan
    edges[-5:, :] = np.nan
    edges[:, :5] = np.nan
    edges[:, -5:] = np.nan
    overlay = edges

    # abs_max = np.percentile(np.abs(data), percentile)
    # abs_min = abs_max

    # plt.pcolormesh(range(mask.shape[0]), range(mask.shape[1]), mask, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    plt.imshow(mask, extent=extent, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    plt.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    plt.axis('off')
    plt.title('attention area respecting latent rule')
    plt.show()

    # plt.figure(figsize=(12, 4))
    # for i in range(latent_dim):
    #     img2show, mask = exp.get_image_rule(features=[i], samples=10)
    #     plt.subplot(1, 4, i+1)
    #     if use_rgb:
    #         plt.imshow(img2show)
    #     else:
    #         plt.imshow(img2show, cmap='gray')
    #     plt.pcolormesh(range(mask.shape[0]), range(mask.shape[1]), mask, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    #     plt.title('varying dim %d' % i)
    # plt.suptitle('attention area respecting latent rule')
    # plt.show()
    #
    # prototypes = exp.get_prototypes_respecting_rule(num_prototypes=5, eps=255*0.25)
    # for pimg in prototypes:
    #     bbo = bb_predict(np.array([gray2rgb(pimg)]))[0]
    #     if use_rgb:
    #         plt.imshow(pimg)
    #     else:
    #         plt.imshow(pimg, cmap='gray')
    #     plt.title('prototype %s' % bbo)
    #     plt.show()
    #
    # prototypes, diff_list = exp.get_prototypes_respecting_rule(num_prototypes=5, return_diff=True)
    # for pimg, diff in zip(prototypes, diff_list):
    #     bbo = bb_predict(np.array([gray2rgb(pimg)]))[0]
    #     plt.subplot(1, 2, 1)
    #     if use_rgb:
    #         plt.imshow(pimg)
    #     else:
    #         plt.imshow(pimg, cmap='gray')
    #     plt.title('prototype %s' % bbo)
    #     plt.subplot(1, 2, 2)
    #     plt.title('differences')
    #     if use_rgb:
    #         plt.imshow(pimg)
    #     else:
    #         plt.imshow(pimg, cmap='gray')
    #     plt.pcolormesh(range(diff.shape[0]), range(diff.shape[1]), diff, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    #     plt.show()
    #
    # cprototypes = exp.get_counterfactual_prototypes(eps=0.01)
    # for cpimg in cprototypes:
    #     bboc = bb_predict(np.array([cpimg]))[0]
    #     if use_rgb:
    #         plt.imshow(cpimg)
    #     else:
    #         plt.imshow(cpimg, cmap='gray')
    #     plt.title('cf - black box %s' % bboc)
    #     plt.show()
    #
    # cprototypes_interp = exp.get_counterfactual_prototypes(eps=0.01, interp=5)
    # for cpimg_interp in cprototypes_interp:
    #     for i, cpimg in enumerate(cpimg_interp):
    #         bboc = bb_predict(np.array([cpimg]))[0]
    #         plt.subplot(1, 5, i+1)
    #         if use_rgb:
    #             plt.imshow(cpimg)
    #         else:
    #             plt.imshow(cpimg, cmap='gray')
    #         plt.title('%s' % bboc)
    #     fo = bb_predict(np.array([cpimg_interp[0]]))[0]
    #     to = bb_predict(np.array([cpimg_interp[-1]]))[0]
    #     plt.suptitle('black box - from %s to %s' % (fo, to))
    #     plt.show()


if __name__ == "__main__":
    main()
