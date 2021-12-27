from xailib.models.bbox import AbstractBBox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import torch

class ImageInsDel():
    def __init__(self, predict, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.

        Args:
            predict (func): function that takes in input a numpy array and return the prediction.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
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