import random
import pickle
import numpy as np

from abc import abstractmethod
from deap import base, creator, tools, algorithms

from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist, hamming, cosine

from ..ilore.util import sigmoid, neuclidean

import warnings

warnings.filterwarnings("ignore")


class ImageNeighborhoodGenerator(object):

    def __init__(self, bb_predict, ocr=0.1):
        self.bb_predict = bb_predict
        self.ocr = ocr  # other class ratio

    @abstractmethod
    def generate(self, img, num_samples=1000):
        return


class ImageAdversarialGeneratorLatent(ImageNeighborhoodGenerator):

    def __init__(self, bb_predict, ocr=0.1, autoencoder=None, min_width=1, min_height=1,
                 scale=False, valid_thr=0.5):
        super(ImageAdversarialGeneratorLatent, self).__init__(bb_predict, ocr)
        self.autoencoder = autoencoder
        self.min_width = min_width
        self.min_height = min_height
        self.scale = scale
        self.valid_thr = valid_thr

    @abstractmethod
    def generate(self, img, num_samples=1000):
        return

    def generate_latent(self):
        #while True:
        lz_img = np.random.normal(size=(1, self.autoencoder.latent_dim))
        #    if self.autoencoder.discriminator is not None and self.valid_thr > 0.0:
        #        discriminator_out = self.autoencoder.discriminator.predict(lz_img)[0][0]
        #        if discriminator_out > self.valid_thr:
        #            return lz_img
        #    else:
        return lz_img

    def generate_latent_samples(self, num_samples):
        lZ_img = list()
        for i in range(num_samples):
            lz_img = self.generate_latent()
            lZ_img.append(lz_img)

        lZ_img = np.array(lZ_img).reshape((num_samples, self.autoencoder.latent_dim))
        return lZ_img

    def _balance_neigh(self, Z, Z_img, Yb, num_samples, class_value):
        class_counts = np.unique(Yb, return_counts=True)
        if len(class_counts[0]) <= 2:
            ocs = int(np.round(num_samples * self.ocr))
            Z1, Z1_img = self._rndgen_not_class(ocs, class_value)
            if len(Z1) > 0:
                Z = np.concatenate((Z, Z1), axis=0)
                Z_img = np.concatenate((Z_img, Z1_img), axis=0)
        else:
            max_cc = np.max(class_counts[1])
            max_cc2 = np.max([cc for cc in class_counts[1] if cc != max_cc])
            if max_cc2 / len(Yb) < self.ocr:
                ocs = int(np.round(num_samples * self.ocr)) - max_cc2
                Z1, Z1_img = self._rndgen_not_class(ocs, class_value)
                if len(Z1) > 0:
                    Z = np.concatenate((Z, Z1), axis=0)
                    Z_img = np.concatenate((Z_img, Z1_img), axis=0)
        return Z, Z_img

    def _rndgen_not_class(self, num_samples, class_value, max_iter=1000):
        lZ_img = list()
        Z_img = list()
        iter_count = 0
        while len(lZ_img) < num_samples:
            lz_img = self.generate_latent()
            z_img = self.autoencoder.decode(lz_img)[0]
            bbo = self.bb_predict(np.array([z_img]))[0]
            if bbo != class_value:
                lZ_img.append(lz_img[0])
                Z_img.append(z_img)
            iter_count += 1
            if iter_count >= max_iter:
                break

        lZ_img = np.array(lZ_img)
        Z_img = np.array(Z_img)
        return lZ_img, Z_img

    def _fix_neigh(self, Z, Z_img, Yb, class_value, ratio=0.25, max_iter=10000):
        class_counts = np.unique(Yb, return_counts=True)
        class_counts = {k: v for k, v in zip(class_counts[0], class_counts[1])}
        class_max = max(class_counts, key=class_counts.get)

        if class_max == class_value:
            # print('not to fix')
            return Z, Z_img, Yb

        # print('fixing')

        class_value_value = class_counts[class_value]
        num_samples = len(Yb)
        missing_class_value = int(num_samples * ratio - class_value_value)
        # missing_class_value = class_counts[class_max] - class_value_value + 1

        Z1 = list()
        Z1_img = list()
        Yb1 = list()
        iter_count = 0

        # print(class_counts)

        while len(Z1) < missing_class_value:
            lz_img = self.generate_latent()
            z_img = self.autoencoder.decode(lz_img)[0]
            bbo = self.bb_predict(np.array([z_img]))[0]
            if bbo == class_value:
                Z1.append(lz_img[0])
                Z1_img.append(z_img)
                Yb1.append(bbo)
            iter_count += 1
            if iter_count >= max_iter:
                break

        Z1 = np.array(Z1)
        Z1_img = np.array(Z1_img)
        Yb1 = np.array(Yb1)

        if len(Z1) > 0:
            Z = np.concatenate((Z, Z1), axis=0)
            Z_img = np.concatenate((Z_img, Z1_img), axis=0)
            Yb = np.concatenate((Yb, Yb1), axis=0)

        # print('fixed', iter_count)

        return Z, Z_img, Yb


class ImageRandomAdversarialGeneratorLatent(ImageAdversarialGeneratorLatent):

    def __init__(self, bb_predict, ocr=0.1, autoencoder=None, min_width=1, min_height=1, scale=False, valid_thr=0.5):
        super(ImageRandomAdversarialGeneratorLatent, self).__init__(bb_predict, ocr, autoencoder, min_width, min_height,
                                                                    scale, valid_thr)

    def generate(self, img, num_samples=1000):

        # generate neighborhood in latent space, decode and label it
        lZ_img = self.generate_latent_samples(num_samples)
        lZ_img[0] = self.autoencoder.encode(np.array([img]))[0]
        # print('Latent space')
        # print(lZ_img[:5])

        Z_img = self.autoencoder.decode(lZ_img)
        Z_img[0] = img.copy()
        # print('Real space')
        # print(Z_img.shape)

        Yb = self.bb_predict(Z_img)
        class_value = Yb[0]

        lZ_img, Z_img = self._balance_neigh(lZ_img, Z_img, Yb, num_samples, class_value)
        Yb = self.bb_predict(Z_img)

        # lZ_img, Z_img, Yb = self._fix_neigh(lZ_img, Z_img, Yb, class_value)

        Z = np.array(lZ_img)

        if self.scale:
            scaler = MinMaxScaler()
            Z = scaler.fit_transform(Z)

        # print('Latent space 2')
        # print(Z[:5])
        # print(Yb[:5])

        return Z, Yb, class_value


class ImageGeneticAdversarialGeneratorLatent(ImageAdversarialGeneratorLatent):

    def __init__(self, bb_predict, ocr=0.1, autoencoder=None, min_width=1, min_height=1, scale=False, valid_thr=0.5,
                 alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, random_seed=None, verbose=False):
        super(ImageGeneticAdversarialGeneratorLatent, self).__init__(bb_predict, ocr, autoencoder, min_width,
                                                                     min_height, scale, valid_thr)

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.metric = metric
        self.ngen = ngen
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.tournsize = tournsize
        self.halloffame_ratio = halloffame_ratio
        self.verbose = verbose
        random.seed(random_seed)

    def generate(self, img, num_samples=1000):

        # generate neighborhood in latent space, decode and label it
        num_samples_eq = int(np.round(num_samples * 0.5))
        num_samples_noteq = int(np.round(num_samples * 0.5))

        limg = self.autoencoder.encode(np.array([img.copy()]))[0]

        toolbox_eq = self.setup_toolbox(limg, self.fitness_equal, num_samples_eq)
        population_eq, halloffame_eq, logbook_eq = self.fit(toolbox_eq, num_samples_eq)
        lZ_eq = self.add_halloffame(population_eq, halloffame_eq)

        toolbox_noteq = self.setup_toolbox(limg, self.fitness_notequal, num_samples_noteq)
        population_noteq, halloffame_noteq, logbook_noteq = self.fit(toolbox_noteq, num_samples_noteq)
        lZ_noteq = self.add_halloffame(population_noteq, halloffame_noteq)

        lZ_img = np.concatenate((lZ_eq, lZ_noteq), axis=0)
        # print('Latent space')
        # print(lZ_img[:5])
        Z_img = self.autoencoder.decode(lZ_img)
        Z_img[0] = img.copy()
        # print('Real space')
        # print(np.unique(Z_img[:5]))

        Yb = self.bb_predict(Z_img)
        class_value = Yb[0]

        Z, Z_img = self._balance_neigh(lZ_img, Z_img, Yb, num_samples, class_value)
        Yb = self.bb_predict(Z_img)
        # print('Interpretable space')
        # print(Z[:5])
        # print(Yb[:5])

        # lZ_img, Z_img, Yb = self._fix_neigh(lZ_img, Z_img, Yb, class_value)
        
        if self.scale:
            scaler = MinMaxScaler()
            Z = scaler.fit_transform(Z)

        return Z, Yb, class_value

    def add_halloffame(self, population, halloffame):
        fitness_values = [p.fitness.wvalues[0] for p in population]
        fitness_values = sorted(fitness_values)
        fitness_diff = [fitness_values[i + 1] - fitness_values[i] for i in range(0, len(fitness_values) - 1)]

        index = np.max(np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist())
        fitness_value_thr = fitness_values[index]

        Z = list()
        for p in population:
            Z.append(p)

        for h in halloffame:
            if h.fitness.wvalues[0] > fitness_value_thr:
                Z.append(h)

        return np.array(Z)

    def setup_toolbox(self, x, evaluate, population_size):
        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", np.ndarray, fitness=creator.fitness)

        toolbox = base.Toolbox()
        toolbox.register("feature_values", self.record_init, x)
        toolbox.register("individual", tools.initIterate, creator.individual, toolbox.feature_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)

        toolbox.register("clone", self.clone)
        toolbox.register("evaluate", evaluate, x)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate, toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

        return toolbox

    def fit(self, toolbox, population_size):
        halloffame_size = int(np.round(population_size * self.halloffame_ratio))

        population = toolbox.population(n=population_size)
        halloffame = tools.HallOfFame(halloffame_size, similar=np.array_equal)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=self.cxpb, mutpb=self.mutpb,
                                                  ngen=self.ngen, stats=stats, halloffame=halloffame,
                                                  verbose=self.verbose)

        return population, halloffame, logbook

    def record_init(self, x):
        return x

    def random_init(self):
        z = self.generate_latent()
        return z

    def clone(self, x):
        return pickle.loads(pickle.dumps(x))

    def mutate(self, toolbox, x):
        #while True:
        mutated = toolbox.clone(x)
        mutation_mask = np.random.choice([False, True], self.autoencoder.latent_dim, p=[1 - self.mutpb, self.mutpb])
        mutator = np.random.normal(size=self.autoencoder.latent_dim)
        mutated[mutation_mask] = mutator[mutation_mask]
        #    if self.autoencoder.discriminator is not None:
        #        discriminator_out = self.autoencoder.discriminator.predict(mutated.reshape(1, -1))[0][0]
        #        if discriminator_out > self.valid_thr:
        #            return mutated,
        #    else:
        return mutated,

    def fitness_equal(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        feature_similarity = sigmoid(feature_similarity_score) if feature_similarity_score < 1.0 else 0.0

        y = self.bb_predict(self.autoencoder.decode(np.array([x])))[0]
        y1 = self.bb_predict(self.autoencoder.decode(np.array([x1])))[0]
        # print(y, y1)

        target_similarity_score = 1.0 - hamming(y, y1)
        target_similarity = sigmoid(target_similarity_score)
        # print(target_similarity_score, target_similarity)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,

    def fitness_notequal(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        feature_similarity = sigmoid(feature_similarity_score)

        y = self.bb_predict(self.autoencoder.decode(np.array([x])))[0]
        y1 = self.bb_predict(self.autoencoder.decode(np.array([x1])))[0]
        # print(y, y1)

        target_similarity_score = 1.0 - hamming(y, y1)
        target_similarity = 1.0 - sigmoid(target_similarity_score)
        # print(target_similarity_score, target_similarity)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,


class ImageRandomGeneticAdversarialGeneratorLatent(ImageGeneticAdversarialGeneratorLatent,
                                                   ImageRandomAdversarialGeneratorLatent):

    def __init__(self, bb_predict, ocr=0.1, autoencoder=None, min_width=1, min_height=1, scale=False, valid_thr=0.5,
                 alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, random_seed=None, verbose=False):
        super(ImageRandomGeneticAdversarialGeneratorLatent, self).__init__(bb_predict, ocr, autoencoder,
                                                                           min_width, min_height, scale, valid_thr)
        super(ImageRandomGeneticAdversarialGeneratorLatent, self).__init__(bb_predict, ocr, autoencoder,
                                                                           min_width, min_height, scale, valid_thr,
                                                                           alpha1, alpha2, metric, ngen, mutpb, cxpb,
                                                                           tournsize, halloffame_ratio, random_seed,
                                                                           verbose)

    def generate(self, img, num_samples=1000):
        Zr, Ybr, class_value = ImageRandomAdversarialGeneratorLatent.generate(self, img, num_samples // 2)
        Zg, Ybg, _ = ImageGeneticAdversarialGeneratorLatent.generate(self, img, num_samples // 2)
        Z = np.concatenate((Zr, Zg[1:]), axis=0)
        Yb = np.concatenate((Ybr, Ybg[1:]), axis=0)
        return Z, Yb, class_value


class ImageProbaGeneticAdversarialGeneratorLatent(ImageGeneticAdversarialGeneratorLatent):

    def __init__(self, bb_predict, ocr=0.1, autoencoder=None, min_width=1, min_height=1, scale=False, valid_thr=0.5,
                 alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, bb_predict_proba=None, random_seed=None, verbose=False):
        super(ImageProbaGeneticAdversarialGeneratorLatent, self).__init__(bb_predict, ocr, autoencoder,
                                                                          min_width, min_height, scale, valid_thr,
                                                                          alpha1, alpha2, metric, ngen, mutpb, cxpb,
                                                                          tournsize, halloffame_ratio, random_seed,
                                                                          verbose)
        self.bb_predict_proba = bb_predict_proba

    def fitness_equal(self, x, x1):
        return self.fitness_equal_proba(x, x1)

    def fitness_notequal(self, x, x1):
        return self.fitness_notequal_proba(x, x1)

    def fitness_equal_proba(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        feature_similarity = sigmoid(feature_similarity_score) if feature_similarity_score < 1.0 else 0.0
        # feature_similarity = sigmoid(feature_similarity_score)
        # print(feature_similarity_score, feature_similarity)

        y = self.bb_predict_proba(self.autoencoder.decode(np.array([x])))[0]
        y1 = self.bb_predict_proba(self.autoencoder.decode(np.array([x1])))[0]
        # print(self.bb_predict_proba(self._img2bb(x))[0])
        # print(self.bb_predict_proba(self._img2bb(x1))[0])

        target_similarity_score = 1.0 - cosine(y, y1)
        target_similarity = sigmoid(target_similarity_score)
        # print(target_similarity_score, target_similarity)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        # print(evaluation)
        return evaluation,

    def fitness_notequal_proba(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        feature_similarity = sigmoid(feature_similarity_score)

        y = self.bb_predict_proba(self.autoencoder.decode(np.array([x])))[0]
        y1 = self.bb_predict_proba(self.autoencoder.decode(np.array([x1])))[0]
        # print(y, y1)

        target_similarity_score = 1.0 - cosine(y, y1)
        target_similarity = 1.0 - sigmoid(target_similarity_score)
        # print(target_similarity_score, target_similarity)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,


class ImageRandomProbaGeneticAdversarialGeneratorLatent(ImageProbaGeneticAdversarialGeneratorLatent,
                                                        ImageRandomAdversarialGeneratorLatent):

    def __init__(self, bb_predict, ocr=0.1, autoencoder=None,
                 min_width=1, min_height=1, scale=False, valid_thr=0.5,
                 alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, bb_predict_proba=None, random_seed=None, verbose=False):
        super(ImageRandomProbaGeneticAdversarialGeneratorLatent, self).__init__(bb_predict, ocr, autoencoder,
                                                                                min_width, min_height, scale, valid_thr)
        super(ImageRandomProbaGeneticAdversarialGeneratorLatent, self).__init__(bb_predict, ocr, autoencoder,
                                                                                min_width, min_height, scale, valid_thr,
                                                                                alpha1, alpha2, metric, ngen, mutpb,
                                                                                cxpb, tournsize, halloffame_ratio,
                                                                                bb_predict_proba, random_seed, verbose)

    def generate(self, img, num_samples=1000):
        Zr, Ybr, class_value = ImageRandomAdversarialGeneratorLatent.generate(self, img, num_samples // 2)
        Zg, Ybg, _ = ImageProbaGeneticAdversarialGeneratorLatent.generate(self, img, num_samples // 2)
        Z = np.concatenate((Zr, Zg[1:]), axis=0)
        Yb = np.concatenate((Ybr, Ybg[1:]), axis=0)
        return Z, Yb, class_value

