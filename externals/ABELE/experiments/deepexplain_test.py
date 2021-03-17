import numpy as np
import matplotlib.pyplot as plt

from skimage import feature, transform

from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
from deepexplain.tensorflow import DeepExplain


from experiments.exputil import get_dataset
from experiments.exputil import get_black_box


def plot(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    axis.axis('off')
    return axis


def main():

    dataset = 'mnist'
    black_box = 'DNN'
    num_classes = 10

    path = './'
    path_models = path + 'models/'

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)

    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)
    bb, transform = get_black_box(black_box, black_box_filename, use_rgb, return_model=True)
    bb_predict, _ = get_black_box(black_box, black_box_filename, use_rgb)

    i2e = 1
    img = X_test[i2e]
    bbo = bb_predict(np.array([img]))

    with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
        # Need to reconstruct the graph in DeepExplain context, using the same weights.
        # With Keras this is very easy:
        # 1. Get the input tensor to the original model
        input_tensor = bb.layers[0].input
        # print(input_tensor)

        # 2. We now target the output of the last dense layer (pre-softmax)
        # To do so, create a new model sharing the same layers untill the last dense (index -2)
        fModel = Model(inputs=input_tensor, outputs=bb.layers[-2].output)
        target_tensor = fModel(input_tensor)
        # print(target_tensor)

        # print(fModel.summary())

        xs = transform(np.array([img]))
        xs = xs.astype(float)
        print(xs.shape, xs.dtype)
        # xs = X_test[0:10]
        # xs = np.array([rgb2gray(x) for x in xs])
        ys = to_categorical(bbo, num_classes)
        print(len(xs), len(ys), xs.shape, ys.shape)

        attributions = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
        # attributions = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
        # attributions = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys)
        # attributions    = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
        # attributions  = de.explain('elrp', target_tensor, input_tensor, xs, ys=ys)
        # attributions   = de.explain('occlusion', target_tensor, input_tensor, xs, ys=ys)

        # Compare Gradient * Input with approximate Shapley Values
        # Note1: Shapley Value sampling with 100 samples per feature (78400 runs) takes a couple of minutes on a GPU.
        # Note2: 100 samples are not enough for convergence, the result might be affected by sampling variance
        # attributions = de.explain('shapley_sampling', target_tensor, input_tensor, xs, ys=ys, samples=10)

    # print(attributions_gradin)
    # print(attributions_sal)
    # print(attributions_sal.shape)
    plot(attributions[0], xi=xs[0], cmap=plt.cm.BrBG)
    plt.show()

    # relevancy = np.mean(attributions_ig[0], 2)
    # print(relevancy.shape)
    # print(np.unique(relevancy, return_counts=True))
    # print(np.percentile(relevancy, 25))
    # print(np.percentile(relevancy, 50))
    # print(np.percentile(relevancy, 75))


if __name__ == "__main__":
    main()
