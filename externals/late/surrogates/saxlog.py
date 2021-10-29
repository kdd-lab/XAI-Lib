from sktime.classification.shapelet_based import MrSEQLClassifier
from sklearn.linear_model import LogisticRegression
from utils import convert_numpy_to_sktime
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from late.surrogates.sbgdt import plot_binary_heatmap
import matplotlib
import sys
from keras.utils import to_categorical


def plot_factual_and_counterfactual2(
        saxlog_model,
        x,
        x_label,
        draw_on=None,
        figsize=(10, 3),
        fontsize=18,
        labelfontsize=12,
        dpi=60,
        loc=None,
        frameon=False
):
    """if custom_coef:
        coefs = saxlog_model.classifier.coef_
        if len(np.unique(saxlog_model.y)) == 2:
            coefs = np.concatenate([-saxlog_model.classifier.coef_, saxlog_model.classifier.coef_])

        aligned_coefs = list()
        x_sklearn = convert_numpy_to_sktime(x)
        mr_seqs = saxlog_model.seql_model._transform_time_series(x_sklearn)
        x_transformed = saxlog_model.seql_model._to_feature_space(mr_seqs).ravel()
        x = x.ravel()
        for idx_word in range(len(x_transformed)):
            if x_transformed[idx_word] == 0:
                continue
            dummy_ts = np.repeat(np.nan, len(x))
            start_idx, end_idx, feature = map_word_idx_to_ts(x, idx_word, saxlog_model.seql_model)
            if end_idx == len(x):
                end_idx -= 1
            dummy_ts[start_idx:end_idx + 1] = coefs[x_label][idx_word]
            aligned_coefs.append(dummy_ts)
        aligned_coefs_mean = np.nanmean(aligned_coefs, axis=0)
        minima = np.nanmin(coefs[x_label])
        maxima = np.nanmax(coefs[x_label])"""
    x = x.ravel()
    aligned_coefs_mean = saxlog_model.seql_model.map_sax_model(x)[x_label]
    minima = np.nanmin(saxlog_model.seql_model.ots_clf.coef_[x_label])
    maxima = np.nanmax(saxlog_model.seql_model.ots_clf.coef_[x_label])

    # these are here to avoid error in case there aren't values under or over 0 (for DivergingNorm)
    if minima == 0:
        minima -= sys.float_info.epsilon
    if maxima == 0:
        maxima += sys.float_info.epsilon

    norm = matplotlib.colors.DivergingNorm(vmin=minima, vcenter=0, vmax=maxima)

    div = [[norm(minima), "#d62728"], [norm(0), "white"], [norm(maxima), "#2ca02c"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", div)

    title = (r"$b(x)$" + " = " + saxlog_model.labels[x_label] if saxlog_model.labels else r"$b(x)$" + " = " + str(x_label))
    legend_label = r"$x$"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title(title, fontsize=fontsize)
    ax.plot(x.ravel().T if draw_on is None else draw_on.T, c="gray", alpha=0.2, lw=3, label=legend_label)
    ax.pcolorfast((0, len(x) - 1),
                  ax.get_ylim(),
                  norm(aligned_coefs_mean[np.newaxis]),
                  cmap=cmap,
                  alpha=1,
                  vmin=0,
                  vmax=1
                  )
    plt.ylabel("value", fontsize=fontsize)
    plt.xlabel("time-steps", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.show()
    plt.show()


class Saxlog(object):
    def __init__(
            self,
            labels=None,
            random_state=None,
            custom_config=None,
    ):
        self.labels = labels
        self.random_state = random_state
        self.custom_config = custom_config

        self.X = None
        self.X_transformed = None
        self.y = None

        self.subsequence_dictionary = None
        self.name_dictionary = None
        self.subsequences_norm_same_length = None

    def fit(self, X, y):
        #assert len(np.unique(y)) == 2
        self.X = X
        self.y = y
        X = convert_numpy_to_sktime(X)
        seql_model = MrSEQLClassifier(seql_mode='fs', symrep='sax', custom_config=self.custom_config)
        seql_model.fit(X, y)
        mr_seqs = seql_model._transform_time_series(X)
        X_transformed = seql_model._to_feature_space(mr_seqs)

        clf = LogisticRegression(multi_class="multinomial")
        clf.fit(X_transformed, y)

        self.X_transformed = X_transformed
        self.classifier = clf
        self.seql_model = seql_model
        return self

    def predict(self, X):
        X = convert_numpy_to_sktime(X)
        mr_seqs = self.seql_model._transform_time_series(X)
        X_transformed = self.seql_model._to_feature_space(mr_seqs)
        y = self.classifier.predict(X_transformed)
        return y

    def predict_proba(self, X):
        return to_categorical(self.predict(X), num_classes=len(np.unique(self.y)))

    def score(self, X, y):
        return accuracy_score(self.predict(X), y)


    def plot_factual_and_counterfactual(
            self,
            x,
            x_label,
            **kwargs
    ):
        plot_factual_and_counterfactual2(self, x, x_label, **kwargs)
        return self

    def plot_binary_heatmap(self, x_label, **kwargs):
        plot_binary_heatmap(x_label, self.y, self.X_transformed, **kwargs)
        return self


if __name__ == "__main__":
    from late.datasets.datasets import build_cbf

    random_state = 0
    np.random.seed(0)
    dataset_name = "cbf"

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_cbf(random_state=random_state)


    clf = Saxlog(random_state=np.random.seed(0))
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    i = 2

    clf.plot_factual_and_counterfactual(X_train[i:i + 1], y_train[i], custom_coef=False)
    clf.plot_factual_and_counterfactual(X_train[i:i + 1], y_train[i], custom_coef=True)