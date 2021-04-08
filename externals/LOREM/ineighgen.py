import numpy as np

from abc import abstractmethod

from skimage.filters import sobel
from skimage.color import rgb2gray, gray2rgb


class ImageNeighborhoodGenerator(object):

    def __init__(self, bb_predict, ocr=0.1):
        self.bb_predict = bb_predict
        self.ocr = ocr  # other class ratio

    @abstractmethod
    def generate(self, img, num_samples=1000):
        return


class ImageLimeGenerator(ImageNeighborhoodGenerator):

    def __init__(self, bb_predict, ocr=0.1, segmentation_fn=None):
        super(ImageLimeGenerator, self).__init__(bb_predict, ocr)

        self.segmentation_fn = segmentation_fn
        self.segments = None

    def generate(self, img, num_samples=1000, hide_color=None):

        segments = self.segmentation_fn(img)
        segments_list = list(np.unique(segments))

        nbr_features = len(segments_list)

        x0 = img.copy()
        if hide_color is None:
            for s in segments_list:
                x0[segments == s] = (
                    np.mean(img[segments == s][:, 0]),
                    np.mean(img[segments == s][:, 1]),
                    np.mean(img[segments == s][:, 2]))
        else:
            x0[:] = hide_color

        Z = np.random.randint(0, 2, num_samples * nbr_features).reshape((num_samples, nbr_features))

        Z[0, :] = 1
        Z_img = list()
        for z in Z:
            z_img = self.__lime_z2zimg(z, img, x0, segments)
            Z_img.append(z_img)
        Yb = self.bb_predict(Z_img)
        class_value = Yb[0]

        Z, Z_img = self.__balance_neigh(img, Z, Z_img, Yb, num_samples, class_value, nbr_features, x0, segments)
        Yb = self.bb_predict(Z_img)

        return Z, Yb, class_value, segments

    def __lime_z2zimg(self, z, img, x0, segments):
        z_img = np.copy(img)
        zeros = np.where(z == 0)[0]
        mask = np.zeros(segments.shape).astype(bool)
        for z in zeros:
            mask[segments == z] = True
        z_img[mask] = x0[mask]
        return z_img

    def __balance_neigh(self, img, Z, Z_img, Yb, num_samples, class_value, nbr_features, x0, segments):
        class_counts = np.unique(Yb, return_counts=True)
        if len(class_counts[0]) <= 2:
            ocs = int(np.round(num_samples * self.ocr))
            Z1, Z1_img = self.__rndgen_not_class(img, ocs, class_value, nbr_features, x0, segments)
            if len(Z1) > 0:
                Z = np.concatenate((Z, Z1), axis=0)
                Z_img.extend(Z1_img)
        else:
            max_cc = np.max(class_counts[1])
            max_cc2 = np.max([cc for cc in class_counts[1] if cc != max_cc])
            if max_cc2 / len(Yb) < self.ocr:
                ocs = int(np.round(num_samples * self.ocr)) - max_cc2
                Z1, Z1_img = self.__rndgen_not_class(img, ocs, class_value, nbr_features, x0, segments)
                if len(Z1) > 0:
                    Z = np.concatenate((Z, Z1), axis=0)
                    Z_img.extend(Z1_img)
        return Z, Z_img

    def __rndgen_not_class(self, img, num_samples, class_value, nbr_features, x0, segments, max_iter=1000):
        Z = list()
        Z_img = list()
        iter_count = 0
        while len(Z) < num_samples:
            z = np.random.randint(0, 2, nbr_features)
            z_img = self.__lime_z2zimg(z, img, x0, segments)
            if self.bb_predict([z_img])[0] != class_value:
                Z.append(z)
                Z_img.append(z_img)
            iter_count += 1
            if iter_count >= max_iter:
                break

        Z = np.array(Z)
        return Z, Z_img
