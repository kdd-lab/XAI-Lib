import pickle
import numpy as np

from abc import abstractmethod
from scipy.spatial.distance import cdist, hamming, cosine

import random
from deap import base, creator, tools, algorithms
from externals.LOREM.util import sigmoid, calculate_feature_values, neuclidean

import warnings

warnings.filterwarnings("ignore")


class NeighborhoodGenerator(object):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index, ocr=0.1):
        self.bb_predict = bb_predict
        self.feature_values = feature_values
        self.features_map = features_map
        self.nbr_features = nbr_features
        self.nbr_real_features = nbr_real_features
        self.numeric_columns_index = numeric_columns_index
        self.ocr = ocr  # other class ratio

    @abstractmethod
    def generate(self, x, num_samples=1000):
        return

    def generate_synthetic_instance(self, from_z=None, mutpb=1.0):
        z = np.zeros(self.nbr_features) if from_z is None else from_z
        for i in range(self.nbr_real_features):
            if np.random.random() <= mutpb:
                real_feature_value = np.random.choice(self.feature_values[i], size=1, replace=True)
                if i in self.numeric_columns_index:
                    z[i] = real_feature_value
                else:
                    idx = self.features_map[i][real_feature_value[0]]
                    z[idx] = 1.0
        return z

    def balance_neigh(self, x, Z, num_samples):
        Yb = self.bb_predict(Z)
        class_counts = np.unique(Yb, return_counts=True)

        if len(class_counts[0]) <= 2:
            ocs = int(np.round(num_samples * self.ocr))
            Z1 = self.__rndgen_not_class(ocs, self.bb_predict(x.reshape(1, -1))[0])
            if len(Z1) > 0:
                Z = np.concatenate((Z, Z1), axis=0)
        else:
            max_cc = np.max(class_counts[1])
            max_cc2 = np.max([cc for cc in class_counts[1] if cc != max_cc])
            if max_cc2 / len(Yb) < self.ocr:
                ocs = int(np.round(num_samples * self.ocr)) - max_cc2
                Z1 = self.__rndgen_not_class(ocs, self.bb_predict(x.reshape(1, -1))[0])
                if len(Z1) > 0:
                    Z = np.concatenate((Z, Z1), axis=0)
        return Z

    # def __rndgen_class(self, num_samples, class_value, max_iter=1000):
    #     Z = list()
    #     iter_count = 0
    #     while len(Z) < num_samples:
    #         z = self.generate_synthetic_instance()
    #         if self.bb_predict(z.reshape(1, -1))[0] == class_value:
    #             Z.append(z)
    #         iter_count += 1
    #         if iter_count >= max_iter:
    #             break
    #
    #     Z = np.array(Z)
    #     return Z

    def __rndgen_not_class(self, num_samples, class_value, max_iter=1000):
        Z = list()
        iter_count = 0
        multi_label = isinstance(class_value, np.ndarray)
        while len(Z) < num_samples:
            z = self.generate_synthetic_instance()
            y = self.bb_predict(z.reshape(1, -1))[0]
            flag = y != class_value if not multi_label else np.all(y != class_value)
            if flag:
                Z.append(z)
            iter_count += 1
            if iter_count >= max_iter:
                break

        Z = np.array(Z)
        return Z


class RandomGenerator(NeighborhoodGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index, ocr=0.1):
        super(RandomGenerator, self).__init__(bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                                              numeric_columns_index, ocr)

    def generate(self, x, num_samples=1000):
        Z = np.zeros((num_samples, self.nbr_features))
        for j in range(num_samples):
            Z[j] = self.generate_synthetic_instance()

        Z = super(RandomGenerator, self).balance_neigh(x, Z, num_samples)
        Z[0] = x.copy()
        return Z


class GeneticGenerator(NeighborhoodGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features, numeric_columns_index,
                 ocr=0.1, alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, random_seed=None, verbose=False):
        super(GeneticGenerator, self).__init__(bb_predict, feature_values, features_map, nbr_features,
                                               nbr_real_features, numeric_columns_index, ocr)
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

    def generate(self, x, num_samples=1000, return_logbooks=False):
        num_samples_eq = int(np.round(num_samples * 0.5))
        num_samples_noteq = int(np.round(num_samples * 0.5))

        toolbox_eq = self.setup_toolbox(x, self.fitness_equal, num_samples_eq)
        population_eq, halloffame_eq, logbook_eq = self.fit(toolbox_eq, num_samples_eq)
        Z_eq = self.add_halloffame(population_eq, halloffame_eq)
        # print(logbook_eq)

        # rndgen = RandomGenerator(self.bb)
        # R = rndgen.rndgen_not_class(x, self.feature_values, num_samples, self.bb.predict(x.reshape(1, -1))[0])
        # Rn = (R - np.min(R)) / (np.max(R) - np.min(R))
        # distances = cdist(Rn, Rn[0].reshape(1, -1), metric=self.metric).ravel()
        # x1 = R[np.argsort(distances)[0]]

        toolbox_noteq = self.setup_toolbox(x, self.fitness_notequal, num_samples_noteq)
        # toolbox_noteq = self.setup_toolbox_noteq(x1, x1, self.fitness_notequal, num_samples_noteq)
        population_noteq, halloffame_noteq, logbook_noteq = self.fit(toolbox_noteq, num_samples_noteq)
        Z_noteq = self.add_halloffame(population_noteq, halloffame_noteq)
        # print(logbook_noteq)

        Z = np.concatenate((Z_eq, Z_noteq), axis=0)

        Z = super(GeneticGenerator, self).balance_neigh(x, Z, num_samples)
        Z[0] = x.copy()

        if return_logbooks:
            return Z, [logbook_eq, logbook_noteq]

        return Z

    def add_halloffame(self, population, halloffame):
        fitness_values = [p.fitness.wvalues[0] for p in population]
        fitness_values = sorted(fitness_values)
        fitness_diff = [fitness_values[i + 1] - fitness_values[i] for i in range(0, len(fitness_values) - 1)]

        sorted_array = np.argwhere(fitness_diff == np.amax(fitness_diff)).flatten().tolist()
        if len(sorted_array) == 0:
            fitness_value_thr = -np.inf
        else:
            index = np.max(sorted_array)
            fitness_value_thr = fitness_values[index]

        Z = list()
        for p in population:
            # if p.fitness.wvalues[0] > fitness_value_thr:
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

    def setup_toolbox_noteq(self, x, x1, evaluate, population_size):

        creator.create("fitness", base.Fitness, weights=(1.0,))
        creator.create("individual", np.ndarray, fitness=creator.fitness)

        toolbox = base.Toolbox()
        toolbox.register("feature_values", self.record_init, x1)
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
        z = self.generate_synthetic_instance()
        return z

    def clone(self, x):
        return pickle.loads(pickle.dumps(x))

    def mutate(self, toolbox, x):
        z = toolbox.clone(x)
        # for i in range(self.nbr_features):
        #         #     if np.random.random() <= self.mutpb:
        #         #         z[i] = np.random.choice(self.feature_values[i], size=1, replace=True)
        z = self.generate_synthetic_instance(from_z=z, mutpb=self.mutpb)
        return z,

    def fitness_equal(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        # feature_similarity = feature_similarity_score if feature_similarity_score >= self.eta1 else 0.0
        feature_similarity = sigmoid(feature_similarity_score) if feature_similarity_score < 1.0 else 0.0
        # feature_similarity = sigmoid(feature_similarity_score)

        y = self.bb_predict(x.reshape(1, -1))[0]
        y1 = self.bb_predict(x1.reshape(1, -1))[0]

        target_similarity_score = 1.0 - hamming(y, y1)
        # target_similarity = target_similarity_score if target_similarity_score >= self.eta2 else 0.0
        target_similarity = sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,

    def fitness_notequal(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        # feature_similarity = feature_similarity_score if feature_similarity_score >= self.eta1 else 0.0
        feature_similarity = sigmoid(feature_similarity_score)

        y = self.bb_predict(x.reshape(1, -1))[0]
        y1 = self.bb_predict(x1.reshape(1, -1))[0]

        target_similarity_score = 1.0 - hamming(y, y1)
        # target_similarity = target_similarity_score if target_similarity_score < self.eta2 else 0.0
        target_similarity = 1.0 - sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,


class GeneticProbaGenerator(GeneticGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index,
                 ocr=0.1, alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2,
                 cxpb=0.5, tournsize=3, halloffame_ratio=0.1, bb_predict_proba=None, random_seed=None, verbose=False):
        super(GeneticProbaGenerator, self).__init__(bb_predict, feature_values, features_map,
                                                    nbr_features, nbr_real_features, numeric_columns_index,
                                                    ocr, alpha1, alpha2, metric, ngen, mutpb,
                                                    cxpb, tournsize, halloffame_ratio, random_seed,
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

        y = self.bb_predict_proba(x.reshape(1, -1))[0]
        y1 = self.bb_predict_proba(x1.reshape(1, -1))[0]

        # target_similarity_score = np.sum(np.abs(y - y1))
        target_similarity_score = 1.0 - cosine(y, y1)
        target_similarity = sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,

    def fitness_notequal_proba(self, x, x1):
        feature_similarity_score = 1.0 - cdist(x.reshape(1, -1), x1.reshape(1, -1), metric=self.metric).ravel()[0]
        feature_similarity = sigmoid(feature_similarity_score)

        y = self.bb_predict_proba(x.reshape(1, -1))[0]
        y1 = self.bb_predict_proba(x1.reshape(1, -1))[0]

        # target_similarity_score = np.sum(np.abs(y - y1))
        target_similarity_score = 1.0 - cosine(y, y1)
        target_similarity = 1.0 - sigmoid(target_similarity_score)

        evaluation = self.alpha1 * feature_similarity + self.alpha2 * target_similarity
        return evaluation,


class RandomGeneticGenerator(GeneticGenerator, RandomGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features, numeric_columns_index,
                 ocr=0.1, alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2, cxpb=0.5,
                 tournsize=3, halloffame_ratio=0.1, random_seed=None, verbose=False):
        super(RandomGeneticGenerator, self).__init__(bb_predict, feature_values, features_map, nbr_features,
                                                     nbr_real_features, numeric_columns_index, ocr)
        super(RandomGeneticGenerator, self).__init__(bb_predict, feature_values, features_map, nbr_features,
                                                     nbr_real_features, numeric_columns_index, ocr, alpha1,
                                                     alpha2, metric, ngen, mutpb, cxpb, tournsize,
                                                     halloffame_ratio, random_seed, verbose)

    def generate(self, x, num_samples=1000):
        Zg = GeneticGenerator.generate(self, x, num_samples // 2)
        Zr = RandomGenerator.generate(self, x, num_samples // 2)
        Z = np.concatenate((Zg, Zr[1:]), axis=0)
        return Z


class RandomGeneticProbaGenerator(GeneticProbaGenerator, RandomGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index,
                 ocr=0.1, alpha1=0.5, alpha2=0.5, metric=neuclidean, ngen=100, mutpb=0.2,
                 cxpb=0.5, tournsize=3, halloffame_ratio=0.1, bb_predict_proba=None, random_seed=None, verbose=False):
        super(RandomGeneticProbaGenerator, self).__init__(bb_predict, feature_values, features_map,
                                                          nbr_features, nbr_real_features, numeric_columns_index,
                                                          ocr)
        super(RandomGeneticProbaGenerator, self).__init__(bb_predict, feature_values, features_map,
                                                          nbr_features, nbr_real_features, numeric_columns_index,
                                                          ocr, alpha1, alpha2, metric, ngen, mutpb,
                                                          cxpb, tournsize, halloffame_ratio, bb_predict_proba,
                                                          random_seed, verbose)

    def generate(self, x, num_samples=1000):
        Zg = GeneticProbaGenerator.generate(self, x, num_samples // 2)
        Zr = RandomGenerator.generate(self, x, num_samples // 2)
        Z = np.concatenate((Zg, Zr[1:]), axis=0)
        return Z


class ClosestInstancesGenerator(NeighborhoodGenerator):

    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index, ocr=0.1, K=None, rK=None, k=None, core_neigh_type='unified', alphaf=0.5,
                 alphal=0.5, metric_features=neuclidean, metric_labels='hamming', categorical_use_prob=True,
                 continuous_fun_estimation=False, size=1000, verbose=False):
        super(ClosestInstancesGenerator, self).__init__(bb_predict, feature_values, features_map, nbr_features,
                                                        nbr_real_features, numeric_columns_index, ocr)
        self.K = K
        self.rK = rK
        self.k = k if k is not None else int(0.5 * np.sqrt(len(self.rK))) + 1
        # self.k = np.min([self.k, len(self.rK)])
        self.core_neigh_type = core_neigh_type
        self.alphaf = alphaf
        self.alphal = alphal
        self.metric_features = metric_features
        self.metric_labels = metric_labels
        self.categorical_use_prob = categorical_use_prob
        self.continuous_fun_estimation = continuous_fun_estimation
        self.size = size
        self.verbose = verbose

    def generate(self, x, num_samples=1000):

        K = np.concatenate((x.reshape(1, -1), self.K), axis=0)
        Yb = self.bb_predict(K)
        if self.core_neigh_type == 'mixed':
            Kn = (K - np.min(K)) / (np.max(K) - np.min(K))
            fdist = cdist(Kn, Kn[0].reshape(1, -1), metric=self.metric_features).ravel()
            rk_idxs = np.where(np.argsort(fdist)[:max(int(self.k * self.alphaf), 2)] < len(self.rK))[0]
            Zf = self.rK[rk_idxs]

            ldist = cdist(Yb, Yb[0].reshape(1, -1), metric=self.metric_labels).ravel()
            rk_idxs = np.where(np.argsort(ldist)[:max(int(self.k * self.alphal), 2)] < len(self.rK))[0]
            Zl = self.rK[rk_idxs]
            rZ = np.concatenate((Zf, Zl), axis=0)
        elif self.core_neigh_type == 'unified':
            def metric_unified(x, y):
                n = K.shape[1]
                m = Yb.shape[1]
                distf = cdist(x[:n].reshape(1, -1), y[:n].reshape(1, -1), metric=self.metric_features).ravel()
                distl = cdist(x[n:].reshape(1, -1), y[n:].reshape(1, -1), metric=self.metric_labels).ravel()
                return n / (n + m) * distf + m / (n + m) * distl
            U = np.concatenate((K, Yb), axis=1)
            Un = (U - np.min(U)) / (np.max(U) - np.min(U))
            udist = cdist(Un, Un[0].reshape(1, -1), metric=metric_unified).ravel()
            rk_idxs = np.where(np.argsort(udist)[:self.k] < len(self.rK))[0]
            rZ = self.rK[rk_idxs]
        else:  # self.core_neigh_type == 'simple':
            Kn = (K - np.min(K)) / (np.max(K) - np.min(K))
            fdist = cdist(Kn, Kn[0].reshape(1, -1), metric=self.metric_features).ravel()
            rk_idxs = np.where(np.argsort(fdist)[:self.k] < len(self.rK))[0]
            Zf = self.rK[rk_idxs]
            rZ = Zf

        if self.verbose:
            print('calculating feature values')

        feature_values = calculate_feature_values(rZ, self.numeric_columns_index,
                                                  categorical_use_prob=self.categorical_use_prob,
                                                  continuous_fun_estimation=self.continuous_fun_estimation,
                                                  size=self.size)
        rndgen = RandomGenerator(self.bb_predict, feature_values, self.features_map, self.nbr_features,
                                 self.nbr_real_features, self.numeric_columns_index, self.ocr)
        Z = rndgen.generate(x, num_samples)
        return Z


