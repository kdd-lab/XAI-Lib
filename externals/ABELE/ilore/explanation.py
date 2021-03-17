import numpy as np
import matplotlib.pyplot as plt

from skimage.color import gray2rgb
from ..ilore.util import get_knee_point_value


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class Explanation(object):

    def __init__(self):
        self.bb_pred = None
        self.dt_pred = None

        self.rule = None
        self.crules = None
        self.deltas = None

        self.fidelity = None
        self.dt = None
        self.Z = None

    def __str__(self):
        deltas_str = '{ '
        for i, delta in enumerate(self.deltas):
            deltas_str += '      { ' if i > 0 else '{ '
            deltas_str += ', '.join([str(s) for s in delta])
            deltas_str += ' },\n'
        deltas_str = deltas_str[:-2] + ' }'
        return 'r = %s\nc = %s' % (self.rule, deltas_str)

    def rstr(self):
        return self.rule

    def cstr(self):
        deltas_str = '{ '
        for i, delta in enumerate(self.deltas):
            deltas_str += '{ ' if i > 0 else '{ '
            deltas_str += ', '.join([str(s) for s in delta])
            deltas_str += ' } --> %s, ' % self.crules[i]._cstr()
        deltas_str = deltas_str[:-2] + ' }'
        return deltas_str


class ImageExplanation(Explanation):
    def __init__(self, img, autoencoder, bb_predict, neighgen, use_rgb):
        super(ImageExplanation).__init__()
        self.img = img
        self.autoencoder = autoencoder
        self.bb_predict = bb_predict
        self.neighgen = neighgen
        self.use_rgb = use_rgb

        self.limg = self.autoencoder.encode(np.array([img]))[0]

    def get_image_rule(self, features=None, samples=10, for_show=True):

        img2show = np.copy(self.img)
        prototypes, diff_list = self.get_prototypes_respecting_rule(num_prototypes=samples, features=features,
                                                                    return_diff=True)

        diff = np.median(diff_list, axis=0)

        return img2show, diff

    def get_prototypes_respecting_rule(self, num_prototypes=5, return_latent=False, return_diff=False, features=None,
                                       max_attempts=100000):
        img2show = np.copy(self.img)
        timg = rgb2gray(img2show) if not self.use_rgb else img2show

        features = [i for i in range(self.autoencoder.latent_dim)] if features is None else features
        all_features = [i for i in range(self.autoencoder.latent_dim)]

        prototypes = list()
        lprototypes = list()
        diff_masks = list()
        attempts = 0
        while len(prototypes) < num_prototypes and attempts <= max_attempts:
            lpimg = self.limg.copy()
            mutation = self.neighgen.generate_latent()[0]
            attempts += 1
            mutation_mask = [f in features for f in range(self.autoencoder.latent_dim)]
            lpimg[mutation_mask] = mutation[mutation_mask]

            if self.rule.is_covered(lpimg, all_features):
                pimg = self.autoencoder.decode(lpimg.reshape(1, -1))[0]
                bbo = self.bb_predict(np.array([pimg]))[0]
                if bbo == self.bb_pred:
                    pimg = rgb2gray(pimg) if not self.use_rgb else pimg
                    diff = timg - pimg
                    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff)) * 255
                    diff = np.mean(diff, 2) if self.use_rgb else diff
                    # diff = rgb2gray(diff) if self.use_rgb else diff

                    values, counts = np.unique(np.abs(diff), return_counts=True)
                    sorted_counts = sorted(counts, reverse=True)
                    sorted_counts_idx = np.argsort(counts)[::-1]
                    values = values[sorted_counts_idx]
                    idx_knee = get_knee_point_value(np.log(sorted_counts))

                    th_val = values[idx_knee]
                    gap = np.abs(127.5 - th_val)
                    th_val_l = 127.5 - gap
                    th_val_u = 127.5 + gap
                    # print(values[:10])
                    # print(idx_knee, idx_knee2, th_val, th_val_l, th_val_u)
                    # plt.plot(np.log(sorted_counts))
                    # plt.plot(values)
                    # sorted_diffs = sorted(diff.ravel(), reverse=True)
                    # idx_knee = get_knee_point_value(sorted_diffs)
                    # th_val = sorted_diffs[idx_knee]
                    # print(idx_knee, th_val)
                    # plt.plot(sorted_diffs)
                    # plt.show()
                    # print(np.where((th_val_l <= diff) & (diff <= th_val)))
                    diff[np.where((th_val_l <= diff) & (diff <= th_val_u))] = 127.5

                    prototypes.append(gray2rgb(pimg))
                    lprototypes.append(lpimg)
                    diff_masks.append(diff)

        if return_latent and return_diff:
            return prototypes, lprototypes, diff_masks

        if return_latent:
            return prototypes, lprototypes

        if return_diff:
            return prototypes, diff_masks

        return prototypes

    def get_counterfactual_prototypes(self, eps=0.01, interp=0):

        if interp in [0, 1, 2, None, False]:
            cprototypes = list()
            for delta in self.deltas:
                limg_new = self.limg.copy()
                for p in delta:
                    if p.op == '>':
                        limg_new[p.att] = p.thr + eps
                    else:
                        limg_new[p.att] = p.thr - eps

                img_new = self.autoencoder.decode(limg_new.reshape(1, -1))[0]
                cprototypes.append(img_new)

            return cprototypes

        elif interp >= 3:

            cprototypes = list()
            for delta in self.deltas:
                cinterp = [np.copy(self.img)]
                limg_new = self.limg.copy()
                gaps = [0] * self.autoencoder.latent_dim
                for p in delta:
                    if p.op == '>':
                        limg_new[p.att] = p.thr + eps
                    else:
                        limg_new[p.att] = p.thr - eps
                    gaps[p.att] = np.abs((self.limg[p.att] - limg_new[p.att]) / interp)

                final_img_new = self.autoencoder.decode(limg_new.reshape(1, -1))[0]

                for i in range(1, interp - 1):
                    limg_new = self.limg.copy()
                    for p in delta:
                        if p.op == '>':
                            limg_new[p.att] = self.limg[p.att] + gaps[p.att] * i
                        else:
                            limg_new[p.att] = self.limg[p.att] - gaps[p.att] * i

                    img_new = self.autoencoder.decode(limg_new.reshape(1, -1))[0]
                    cinterp.append(img_new)

                cinterp.append(final_img_new)
                cprototypes.append(cinterp)

            return cprototypes

    def get_prototypes_not_respecting_rule(self, num_prototypes=5, return_latent=False, return_diff=False,
                                           features=None, max_attempts=100000):
        img2show = np.copy(self.img)
        timg = rgb2gray(img2show) if not self.use_rgb else img2show

        features = [i for i in range(self.autoencoder.latent_dim)] if features is None else features
        all_features = [i for i in range(self.autoencoder.latent_dim)]

        prototypes = list()
        lprototypes = list()
        diff_masks = list()
        attempts = 0
        while len(prototypes) < num_prototypes and attempts <= max_attempts:
            lpimg = self.limg.copy()
            mutation = self.neighgen.generate_latent()[0]
            attempts += 1
            mutation_mask = [f in features for f in range(self.autoencoder.latent_dim)]
            lpimg[mutation_mask] = mutation[mutation_mask]

            if not self.rule.is_covered(lpimg, all_features):
                pimg = self.autoencoder.decode(lpimg.reshape(1, -1))[0]
                bbo = self.bb_predict(np.array([pimg]))[0]
                if bbo != self.bb_pred:
                    pimg = rgb2gray(pimg) if not self.use_rgb else pimg
                    diff = timg - pimg
                    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff)) * 255
                    diff = np.mean(diff, 2) if self.use_rgb else diff
                    # diff = rgb2gray(diff) if self.use_rgb else diff

                    values, counts = np.unique(np.abs(diff), return_counts=True)
                    sorted_counts = sorted(counts, reverse=True)
                    sorted_counts_idx = np.argsort(counts)[::-1]
                    values = values[sorted_counts_idx]
                    idx_knee = get_knee_point_value(np.log(sorted_counts))

                    th_val = values[idx_knee]
                    gap = np.abs(127.5 - th_val)
                    th_val_l = 127.5 - gap
                    th_val_u = 127.5 + gap

                    diff[np.where((th_val_l <= diff) & (diff <= th_val_u))] = 127.5

                    prototypes.append(gray2rgb(pimg))
                    lprototypes.append(lpimg)
                    diff_masks.append(diff)

        if return_latent and return_diff:
            return prototypes, lprototypes, diff_masks

        if return_latent:
            return prototypes, lprototypes

        if return_diff:
            return prototypes, diff_masks

        return prototypes
