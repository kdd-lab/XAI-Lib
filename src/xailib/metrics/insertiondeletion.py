"""
Insertion and Deletion metrics for evaluating explanation faithfulness.

This module provides the Insertion and Deletion metrics for evaluating
how well an explanation captures the important features of a model's
prediction. These metrics are commonly used for evaluating image
explanations (saliency maps, heatmaps).

The Insertion metric measures how quickly the model's prediction confidence
increases as pixels are inserted in order of saliency (most salient first).

The Deletion metric measures how quickly the model's prediction confidence
decreases as pixels are deleted in order of saliency (most salient first).

Classes:
    ImageInsDel: Insertion and Deletion metric calculator for images.

References:
    Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized Input
    Sampling for Explanation of Black-box Models. BMVC.

Example:
    >>> from xailib.metrics.insertiondeletion import ImageInsDel
    >>> import numpy as np
    >>>
    >>> # Define prediction function
    >>> def predict(img):
    ...     return model.predict(img)
    >>>
    >>> # Create metric instance
    >>> deletion = ImageInsDel(predict, mode='del', step=100, substrate_fn=lambda x: x*0)
    >>> insertion = ImageInsDel(predict, mode='ins', step=100, substrate_fn=lambda x: x*0)
    >>>
    >>> # Compute scores
    >>> del_scores = deletion(image, 224, saliency_map)
    >>> ins_scores = insertion(image, 224, saliency_map)
"""

from xailib.models.bbox import AbstractBBox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import torch


class ImageInsDel():
    """
    Insertion and Deletion metric calculator for image explanations.

    This class computes the Insertion and Deletion metrics for evaluating
    the faithfulness of image explanations (saliency maps). The metrics
    measure how the model's confidence changes as pixels are progressively
    inserted or deleted in order of their saliency.

    A good explanation should:
        - Have high Insertion score (confidence quickly increases as
          salient pixels are inserted)
        - Have low Deletion score (confidence quickly decreases as
          salient pixels are deleted)

    The Area Under the Curve (AUC) of these scores can be used as a
    single summary metric.

    Args:
        predict (callable): Function that takes a numpy array image and
            returns prediction probabilities.
        mode (str): Either 'del' for deletion metric or 'ins' for
            insertion metric.
        step (int): Number of pixels modified per iteration.
        substrate_fn (callable): Function mapping original pixels to
            baseline pixels (e.g., black pixels, blurred pixels).

    Attributes:
        predict: The prediction function.
        mode: The metric mode ('del' or 'ins').
        step: Number of pixels per step.
        substrate_fn: The substrate/baseline function.

    Example:
        >>> # Create deletion metric with black baseline
        >>> deletion = ImageInsDel(
        ...     predict=model.predict,
        ...     mode='del',
        ...     step=224,  # pixels per step
        ...     substrate_fn=lambda x: torch.zeros_like(x)  # black baseline
        ... )
        >>> scores = deletion(image, size=224, explanation=saliency_map)
        >>> auc = np.trapz(scores) / len(scores)
    """

    def __init__(self, predict, mode, step, substrate_fn):
        """
        Create deletion/insertion metric instance.

        Args:
            predict (callable): Function that takes a numpy array and
                returns the prediction probabilities.
            mode (str): 'del' for deletion or 'ins' for insertion.
            step (int): Number of pixels modified per one iteration.
            substrate_fn (callable): A mapping from old pixels to new pixels
                (e.g., blurring function or constant function).
        """
        assert mode in ['del', 'ins']
        self.predict = predict
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def __call__(self, img, size, explanation, rgb=True, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair.

        Args:
            img (np.ndarray): normalized image tensor.
            size (int): size of the image ex:224
            explanation (np.ndarray): saliency map.
            rgb (bool): if the image is rgb or grayscale
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        if rgb:
            CH = 3
        else: 
            CH = 1
        HW = size * size # image area
        pred = torch.tensor(self.predict(img))
        top, c = torch.max(pred, 1)
        c = c[0]
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion metric'
            ylabel = 'Pixels deleted'
            start = torch.tensor(img).clone()
            finish = self.substrate_fn(torch.tensor(img))
        elif self.mode == 'ins':
            title = 'Insertion metric'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(torch.tensor(img))
            finish = torch.tensor(img).clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1)
        for i in range(n_steps+1):
            pred = torch.tensor(self.predict(start.numpy()))
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print('class {}: probability {:.3f}'.format(cl[0][0], float(pr[0][0])))
                print('class {}: probability {:.3f}'.format(cl[0][1], float(pr[0][1])))
            scores[i] = pred[0, c]
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                #tensor_imshow(start[0])
                image = (start[0].detach().cpu().numpy()).astype(int)
                if rgb:
                    plt.imshow(np.stack([image[0,:,:],image[1,:,:],image[2,:,:]],axis=-1))
                else:
                    plt.imshow(image, cmap='gray')

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                #plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(1, CH, HW)[0, :, coords] = finish.cpu().numpy().reshape(1, CH, HW)[0, :, coords]
        return scores