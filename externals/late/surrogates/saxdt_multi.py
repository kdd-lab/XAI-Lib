from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.classification.shapelet_based.mrseql.mrseql import PySAX  # custom fork of sktime
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import graphviz
import scipy
from utils import convert_numpy_to_sktime, sliding_window_distance, compute_medoid
from late.surrogates.tree_utils import (NewTree,
                                        get_root_leaf_path,
                                        get_thresholds_signs,
                                        minimumDistance,
                                        prune_duplicate_leaves)
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import math
import numpy as np


def plot_mts(x, offset=15, color="royalblue", lw=3, alpha=0.5, linestyle='-', label=None):
    increment = 0
    for i in range(x.shape[2]):
        plt.plot((x[:, :, i] + increment).ravel(), color=color, lw=lw, alpha=alpha, linestyle=linestyle, label=label)
        increment += offset


def plot_exemplars_and_counterexemplars(
        Z_tilde,
        y,
        x,
        z_tilde,
        x_label,
        labels=None,
        plot_x=True,
        plot_z_tilde=True,
        legend=False,
        no_axes_labels=False,
        **kwargs
):
    """Plots x, z_tilde; exemplars; counterexemplars
    Parameters
    ----------
    Z_tilde : array of shape (n_samples, n_features)
        latent instances
    y : array of shape (n_samples,)
        latent instances labels
    x : array of shape (n_features,)
        instance to explain
    z_tilde : array of shape (n_features,)
        autoencoded instance to explain
    x_label : int
        instance to explain label
    labels : list of shape (n_classes,), optional (default = None)
        list of classes labels

    Returns
    -------
    self
    """
    exemplars_idxs = np.argwhere(y == x_label).ravel()
    counterexemplars_idxs = np.argwhere(y != x_label).ravel()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    # plt.title("Instance to explain: " + r"$b(x)$" + " = " + labels[x_label] if labels
    #           else "Instance to explain: " + r"$b(x)$" + " = " + str(x_label),
    #           fontsize=kwargs.get("fontsize", 12))
    plt.title(r"$b(x)$" + " = " + labels[x_label] if labels else r"$b(x)$" + " = " + str(x_label),
              fontsize=kwargs.get("fontsize", 12))
    if not no_axes_labels:
        #plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
        plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis='both', which='major', labelsize=kwargs.get("fontsize", 12))
    if plot_x:
        plot_mts(x, color="royalblue", linestyle='-', lw=3, alpha=0.9, label=r"$x$")
    if plot_z_tilde:
        plot_mts(z_tilde, color="orange", linestyle='-', lw=3, alpha=0.9, label=r"$\tilde{z}$")
    if legend:
        plt.legend()
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.show()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    # plt.title("Exemplars: " + r"$b(\tilde{Z}_{=})$" + " = " + labels[x_label] if labels
    #           else "Exemplars: " + r"$b(\tilde{Z}_{=})$" + " = " + str(x_label),
    #           fontsize=kwargs.get("fontsize", 12))

    plt.title("LASTS - " + r"$b(\tilde{Z}_{=})$" + " = " + labels[x_label] if labels
              else r"$b(\tilde{Z}_{=})$" + " = " + str(x_label),
              fontsize=kwargs.get("fontsize", 12))
    if not no_axes_labels:
        #plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
        plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis='both', which='major', labelsize=kwargs.get("fontsize", 12))
    for i in range(len(Z_tilde)):
        if x_label == y[i]:
            plot_mts(Z_tilde[i: i + 1], color="#2ca02c", alpha=kwargs.get("alpha", 0.1))
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.show()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    # plt.title("Counterexemplars: " + r"$b(\tilde{Z}_\neq)$" + " " + r"$\neq$" + " " + labels[x_label] if labels
    #           else "Counterexemplars: " + r"$b(\tilde{Z}_\neq)$" + " " + r"$\neq$" + " " + str(x_label),
    #           fontsize=kwargs.get("fontsize", 12))
    plt.title("LASTS - " + r"$b(\tilde{Z}_\neq)$" + " " + r"$\neq$" + " " + labels[x_label] if labels
              else r"$b(\tilde{Z}_\neq)$" + " " + r"$\neq$" + " " + str(x_label),
              fontsize=kwargs.get("fontsize", 12))
    if not no_axes_labels:
        #plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
        plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis='both', which='major', labelsize=kwargs.get("fontsize", 12))
    for i in range(len(Z_tilde)):
        if x_label != y[i]:
            plot_mts(Z_tilde[i: i + 1], color="#d62728", alpha=kwargs.get("alpha", 0.1))
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.show()


def plot_factual_and_counterfactual2(
        saxdt_model,
        x,
        x_label,
        draw_on=None,
        verbose_explanation=True,
        graphical_explanation=True,
        figsize=(10, 3),
        fontsize=18,
        text_height=0.5,
        c_index=0,
        labelfontsize=12,
        dpi=60,
        loc=None,
        frameon=False,
        plot_dictionary=False,
        print_word=True,
        fixed_contained_subsequences=True,
        enhance_not_contained=False,
        no_axes_labels=False,
        offset=15,
):  # FIXME: check this monstruosity again
    """x_sktime = convert_numpy_to_sktime(x)
    mr_seqs = saxdt_model.seql_model._transform_time_series(x_sktime)
    x_transformed = saxdt_model.seql_model._to_feature_space(mr_seqs)"""

    if saxdt_model.subsequence_dictionary is None:
        saxdt_model.create_dictionaries()
    dtree = saxdt_model.decision_tree_explorable
    leaf_id = saxdt_model.find_leaf_id(x)

    factual = get_root_leaf_path(dtree.nodes[leaf_id])
    factual = get_thresholds_signs(dtree, factual)

    nearest_leaf = minimumDistance(dtree.nodes[0], dtree.nodes[leaf_id])[1]

    counterfactual = get_root_leaf_path(dtree.nodes[nearest_leaf])
    counterfactual = get_thresholds_signs(dtree, counterfactual)

    rules_list = [factual, counterfactual]

    if verbose_explanation:
        print("VERBOSE EXPLANATION")
        for i, rule in enumerate(rules_list):
            print()
            print("RULE" if i == 0 else "COUNTERFACTUAL")
            if i == 0:
                print("real class ==", saxdt_model.labels[x_label] if saxdt_model.labels else x_label)
            print("If", end=" ")
            for j, idx_word in enumerate(rule["features"][:-1]):
                idx_dim = saxdt_model.feature_index[idx_word][0]
                idx_orig = saxdt_model.feature_index[idx_word][1]
                word, _, _ = find_feature(idx_orig, saxdt_model.seql_models[idx_dim].sequences)
                print("the word", "'" + word.decode("utf-8") + "'", "(" + str(idx_word) + ")",
                      "is", rule["thresholds_signs"][j], end="")
                if j != len(rule["features"][:-1]) - 1:
                    print(", and", end=" ")
                else:
                    print(",", end=" ")
            print("then the class is", rule["labels"][-1] if not saxdt_model.labels \
                else saxdt_model.labels[rule["labels"][-1]])

    # if plot_dictionary:
    #     idx_word_list = list()
    #     for i, rule in enumerate(rules_list):
    #         for j, idx_word in enumerate(rule["features"][:-1]):
    #             if idx_word not in idx_word_list:
    #                 plot_subsequence_mapping(saxdt_model.subsequence_dictionary, saxdt_model.name_dictionary, idx_word)
    #             idx_word_list.append(idx_word)

    if graphical_explanation:
        contained_subsequences = dict()
        for i, idx_word in enumerate(rules_list[0]["features"][:-1]):
            threshold_sign = rules_list[0]["thresholds_signs"][i]
            if threshold_sign == "contained":
                idx_dim = saxdt_model.feature_index[idx_word][0]
                idx_orig = saxdt_model.feature_index[idx_word][1]
                start_idx, end_idx, feature = map_word_idx_to_ts(x[:, :, idx_dim], idx_orig,
                                                                 saxdt_model.seql_models[idx_dim])
                if end_idx == len(x[:, :, idx_dim].ravel()):
                    end_idx -= 1
                subsequence = x[:, :, idx_dim].ravel()[start_idx:end_idx + 1]
                contained_subsequences[idx_word] = [subsequence]

        # counterfactual rule applied to a counterfactual z_tilde
        # get all the leave ids
        leave_ids = saxdt_model.decision_tree.apply(saxdt_model.X_transformed)
        # get all record in the counterfactual leaf
        counterfactuals_idxs = np.argwhere(leave_ids == nearest_leaf)
        # choose one counterfactual
        counterfactual_idx = counterfactuals_idxs[c_index][0]
        counterfactual_ts = saxdt_model.X[counterfactual_idx:counterfactual_idx + 1]
        counterfactual_y = saxdt_model.y[counterfactual_idx]
        for i, idx_word in enumerate(rules_list[1]["features"][:-1]):
            threshold_sign = rules_list[1]["thresholds_signs"][i]
            if threshold_sign == "contained":
                idx_dim = saxdt_model.feature_index[idx_word][0]
                idx_orig = saxdt_model.feature_index[idx_word][1]
                start_idx, end_idx, feature = map_word_idx_to_ts(counterfactual_ts[:,:,idx_dim], idx_orig,
                                                                 saxdt_model.seql_models[idx_dim])
                if end_idx == len(counterfactual_ts[:, :, idx_dim].ravel()):
                    end_idx -= 1
                subsequence = counterfactual_ts[:, :, idx_dim].ravel()[start_idx:end_idx + 1]
                if idx_word not in contained_subsequences:
                    contained_subsequences[idx_word] = [subsequence]
                else:
                    contained_subsequences[idx_word].append(subsequence)

        title = ("LASTS - Factual Rule " + r"$p_s\rightarrow$" + " " +
                 saxdt_model.labels[rules_list[0]["labels"][-1]] if saxdt_model.labels
                 else "LASTS - Factual Rule " + r"$p_s\rightarrow$" + " " +
                      str(rules_list[0]["labels"][-1]))
        legend_label = r"$x$"
        y_lim = plot_graphical_explanation2(
            saxdt_model=saxdt_model,
            x=x,
            rule=rules_list[0],
            title=title,
            legend_label=legend_label,
            figsize=figsize,
            dpi=dpi,
            fontsize=fontsize,
            text_height=text_height,
            labelfontsize=labelfontsize,
            loc=loc,
            frameon=frameon,
            forced_y_lim=None,
            return_y_lim=True,
            draw_on=draw_on,
            contained_subsequences=contained_subsequences,
            print_word=print_word,
            fixed_contained_subsequences=fixed_contained_subsequences,
            is_factual_for_counterexemplar=False,
            enhance_not_contained=enhance_not_contained,
            no_axes_labels=no_axes_labels,
            offset=offset
        )

        title = ("LASTS - Counterfactual Rule " + r"$q_s\rightarrow$" + " " +
                 saxdt_model.labels[rules_list[1]["labels"][-1]] if saxdt_model.labels
                 else "LASTS - Counterfactual Rule " + r"$q_s\rightarrow$" + " " +
                      str(rules_list[1]["labels"][-1]))
        legend_label = r"$x$"

        plot_graphical_explanation2(
            saxdt_model=saxdt_model,
            x=x,
            rule=rules_list[1],
            title=title,
            legend_label=legend_label,
            figsize=figsize,
            dpi=dpi,
            fontsize=fontsize,
            text_height=text_height,
            labelfontsize=labelfontsize,
            loc=loc,
            frameon=frameon,
            forced_y_lim=None,
            return_y_lim=False,
            draw_on=draw_on,
            contained_subsequences=contained_subsequences,
            print_word=print_word,
            fixed_contained_subsequences=fixed_contained_subsequences,
            is_factual_for_counterexemplar=False,
            enhance_not_contained=enhance_not_contained,
            no_axes_labels=no_axes_labels,
            offset=offset
        )

        print("real class ==", saxdt_model.labels[counterfactual_y] if saxdt_model.labels else counterfactual_y)
        title = ("LASTS - Counterfactual Rule " + r"$q_s\rightarrow$" + " " +
                 saxdt_model.labels[rules_list[1]["labels"][-1]] if saxdt_model.labels
                 else "LASTS - Counterfactual Rule " + r"$q_s\rightarrow$" + " " +
                      str(rules_list[1]["labels"][-1]))
        legend_label = r"$\tilde{z}'$"

        plot_graphical_explanation2(
            saxdt_model=saxdt_model,
            x=counterfactual_ts,
            rule=rules_list[1],
            title=title,
            legend_label=legend_label,
            figsize=figsize,
            dpi=dpi,
            fontsize=fontsize,
            text_height=text_height,
            labelfontsize=labelfontsize,
            loc=loc,
            frameon=frameon,
            forced_y_lim=y_lim,
            return_y_lim=False,
            contained_subsequences=contained_subsequences,
            print_word=print_word,
            fixed_contained_subsequences=fixed_contained_subsequences,
            is_factual_for_counterexemplar=True,
            enhance_not_contained=enhance_not_contained,
            no_axes_labels=no_axes_labels,
            offset=offset
        )


def plot_graphical_explanation2(
        saxdt_model,
        x,
        rule,
        title,
        legend_label,
        figsize,
        dpi,
        fontsize,
        text_height,
        labelfontsize,
        loc,
        frameon,
        is_factual_for_counterexemplar,
        contained_subsequences,
        fixed_contained_subsequences=True,
        forced_y_lim=None,
        return_y_lim=False,
        draw_on=None,
        print_word=True,
        enhance_not_contained=False,
        no_axes_labels=False,
        offset=15
):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.suptitle(title, fontsize=fontsize)
    plot_mts(x if draw_on is None else draw_on, color="royalblue", alpha=0.2, lw=3, label=legend_label, offset=offset)

    for i, idx_word in enumerate(rule["features"][:-1]):
        idx_dim = saxdt_model.feature_index[idx_word][0]
        idx_orig = saxdt_model.feature_index[idx_word][1]
        # feature = saxdt_model.name_dictionary[idx_word]
        threshold_sign = rule["thresholds_signs"][i]
        dummy_ts = np.full_like(x, np.nan)
        if idx_word in contained_subsequences:
            if fixed_contained_subsequences:
                subsequence = contained_subsequences[idx_word][0]
            else:
                if is_factual_for_counterexemplar and (len(contained_subsequences[idx_word]) == 2):
                    subsequence = contained_subsequences[idx_word][1]
                else:
                    subsequence = contained_subsequences[idx_word][0]
        else:
            subsequence = saxdt_model.subsequence_dictionary[idx_word][0].ravel()
            # if enhance_not_contained:
            #     maximum = 0
            #     subseq = None
            #     for subsequence in saxdt_model.subsequence_dictionary[idx_word][:, :, 0]:
            #         dist = sliding_window_euclidean(x.ravel(), subsequence)
            #         if dist > maximum:
            #             maximum = dist
            #             subseq = subsequence
            #     subsequence = subseq
            # else:
            #     subsequence = compute_medoid(saxdt_model.subsequence_dictionary[idx_word][:, :, 0])
        best_alignment_start_idx = sliding_window_distance(x[:, :, idx_dim].ravel(), subsequence)
        best_alignment_end_idx = best_alignment_start_idx + len(subsequence)
        start_idx = best_alignment_start_idx
        end_idx = best_alignment_end_idx
        if end_idx == len(x[:, :, idx_dim].ravel()):
            end_idx -= 1
            subsequence = subsequence[:-1]
        #dummy_ts[:, :, idx_dim][start_idx:end_idx] = subsequence
        dummy_ts[:, start_idx:end_idx, idx_dim] = subsequence
        if threshold_sign == "contained":
            plot_mts(dummy_ts, color="#2ca02c", alpha=0.5, lw=5, label="contained")
            plt.text(
                (start_idx + end_idx) / 2,
                # np.nanmin(dummy_ts) + text_height + ((np.nanmin(dummy_ts) + np.nanmax(dummy_ts))/2),
                text_height + np.mean(subsequence) + (offset*idx_dim),
                str(idx_word),
                fontsize=fontsize - 2,
                c="#2ca02c",
                horizontalalignment='center', verticalalignment='center', weight='bold',
                path_effects=[patheffects.Stroke(linewidth=3, foreground='white', alpha=0.6),
                              patheffects.Normal()]
            )
        else:
            plot_mts(dummy_ts, color="#d62728", alpha=0.5, lw=5, linestyle="--", label="not-contained")
            plt.text(
                (best_alignment_start_idx + best_alignment_end_idx) / 2,
                # np.nanmin(dummy_ts) + text_height + ((np.nanmin(dummy_ts) + np.nanmax(dummy_ts))/2),
                text_height + np.mean(subsequence) + (offset*idx_dim),
                str(idx_word),
                fontsize=fontsize - 2,
                c="#d62728",
                horizontalalignment='center', verticalalignment='center', weight='bold',
                path_effects=[patheffects.Stroke(linewidth=3, foreground='white', alpha=0.6),
                              patheffects.Normal()]
            )
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    if not no_axes_labels:
        plt.xlabel("time-steps", fontsize=fontsize)
        #plt.ylabel("value", fontsize=fontsize)
    plt.legend(
        by_label.values(),
        by_label.keys(),
        frameon=frameon,
        fontsize=labelfontsize,
        loc=loc
    )
    if forced_y_lim is not None:
        plt.gca().set_ylim(forced_y_lim)
    if return_y_lim:
        y_lim = plt.gca().get_ylim()
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.show()
    if return_y_lim:
        return y_lim


def predict_explanation(
        saxdt_model,
        x,
        x_label
):
    if saxdt_model.subsequence_dictionary is None:
        saxdt_model.create_dictionaries()
    dtree = saxdt_model.decision_tree_explorable
    leaf_id = saxdt_model.find_leaf_id(x)

    factual = get_root_leaf_path(dtree.nodes[leaf_id])
    factual = get_thresholds_signs(dtree, factual)

    dummy_ts_x = np.repeat(np.nan, len(x.ravel()))
    dummy_list = list()
    tss = list()
    for i, idx_word in enumerate(factual["features"][:-1]):
        threshold_sign = factual["thresholds_signs"][i]
        if threshold_sign == "contained":
            start_idx, end_idx, _ = map_word_idx_to_ts(x, idx_word, saxdt_model.seql_model)
            if end_idx == len(x.ravel()):
                end_idx -= 1
            dummy_ts_x[start_idx:end_idx + 1] = 1
        else:
            dummy_ts = np.repeat(np.nan, len(x.ravel()))
            #  find other instances with different label wrt x and containing the subsequence
            idxs = np.argwhere((saxdt_model.y != x_label) & (saxdt_model.X_transformed[:, idx_word] == 1)).ravel()
            idx = idxs[0]
            tss.append(saxdt_model.X[idx].ravel())
            start_idx, end_idx, _ = map_word_idx_to_ts(saxdt_model.X[idx: idx + 1], idx_word, saxdt_model.seql_model)
            if end_idx == len(x.ravel()):
                end_idx -= 1
            dummy_ts[start_idx:end_idx + 1] = 1
            dummy_list.append(dummy_ts)
    # if np.nansum(dummy_ts_x) != 0:
    #     dummy_list.append(dummy_ts_x)
    #     tss.append(x.ravel())
    #     last_is_contained = True
    dummy_list.append(dummy_ts_x)
    tss.append(x.ravel())
    dummy_list = np.nan_to_num(np.array(dummy_list))
    tss = np.array(tss)
    return dummy_list, tss


def map_word_idx_to_ts(x, word_idx, seql_model):
    word, cfg_idx, _ = find_feature(word_idx, seql_model.sequences)
    start_idx, end_idx = map_word_to_ts(x, word, seql_model.config[cfg_idx])
    return start_idx, end_idx, word


def map_word_to_ts(x, word, cfg):
    word = [word]
    ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'])
    idx_set = ps.map_patterns(x.ravel(), word)[0]
    # print(idx_set)
    if len(idx_set) == 0:
        return None, None
    idx_set_sorted = sorted(list(idx_set))

    """start_idx = idx_set_sorted[0]
    end_idx = idx_set_sorted[len(idx_set_sorted) - 1]
    for i in range(1, len(idx_set_sorted)):
        #  if index values are not consecutive (i.e. if there is more than one matching index)
        if idx_set_sorted[i] - idx_set_sorted[i - 1] != 1:
            end_idx = idx_set_sorted[i - 1]  # fixme: check the case in which the same word is repeated consecutively
            break
    print("window", cfg['window'], "; ", "subsequence_length", end_idx-start_idx)"""

    start_idx = idx_set_sorted[0]
    end_idx = math.floor(start_idx + (len(word[0]) * cfg['window'] / cfg['word']) - 1)
    if end_idx < len(x) - 1:
        end_idx += 1
    return start_idx, end_idx


def create_subsequences_dictionary(X, X_transformed, n_features, seql_models, feature_index):
    subsequence_dictionary = dict()
    for feature in range(n_features):
        subsequences = list()
        lengths = list()
        for i, x in enumerate(X):
            if X_transformed[i][feature] == 1:
                idx_dim = feature_index[feature][0]
                idx_orig = feature_index[feature][1]
                start_idx, end_idx, feature_string = map_word_idx_to_ts(x[:, idx_dim].ravel(), idx_orig,
                                                                        seql_models[idx_dim])
                if start_idx is not None:
                    subsequence = x[:, idx_dim][start_idx:end_idx]
                    subsequences.append(subsequence)
                    lengths.append(end_idx - start_idx)
        mode = scipy.stats.mode(np.array(lengths))[0][0]
        subsequences_same_length = list()
        for i, subsequence in enumerate(
                subsequences):  # to avoid problems with sequences having slightly different lengths
            if len(subsequence) == mode:
                subsequences_same_length.append(subsequence)
        subsequence_dictionary[feature] = np.array(subsequences_same_length)
    return subsequence_dictionary


def plot_subsequence_mapping(subsequence_dictionary, name_dictionary, feature_idx):
    plt.title(str(feature_idx) + " : " + name_dictionary[feature_idx].decode("utf-8"))
    plt.plot(subsequence_dictionary[feature_idx][:, :, 0].T, c="gray", alpha=0.1)
    # plt.plot(subsequence_dictionary[feature_idx][:,:,0].mean(axis=0).ravel(), c="red")
    plt.plot(compute_medoid(subsequence_dictionary[feature_idx][:, :, 0]), c="red")
    plt.show()


def extract_mapped_subsequences(X, feature_idx, seql_model):
    subsequences = list()
    lengths = []
    for i, x in enumerate(X):
        start_idx, end_idx, feature = map_word_idx_to_ts(x.ravel(), feature_idx, seql_model)
        if start_idx is not None:
            lengths.append(end_idx - start_idx)
            subsequences.append(x.ravel()[start_idx:end_idx])
    same_length_subsequences = list()
    for i, s in enumerate(subsequences):
        if lengths[i] == scipy.stats.mode(np.array(lengths))[0][0]:
            same_length_subsequences.append(s)
    return np.array(same_length_subsequences)


def find_feature(feature_idx, sequences):  # FIXME: check if the count is correct
    idxs = 0
    for i, config in enumerate(sequences):
        if idxs + len(config) - 1 < feature_idx:
            idxs += len(config)
            continue
        elif idxs + len(config) - 1 >= feature_idx:
            j = feature_idx - idxs
            feature = config[j]
            break
    return feature, i, j


"""def find_feature(feature_idx, sequences):
    flat_sequences = list()
    idxs = 0
    for i, sequence in enumerate(sequences):
        if idxs + len(sequence) - 1 < feature_idx:
            idxs += len(sequence)
        flat_sequences.extend(sequence)
    feature = flat_sequences[feature_idx]
    return feature, i, None"""


def map_word_to_window(word_length, window_size):
    mapping = list()
    paa_size = window_size / word_length
    for i in range(word_length):
        windowStartIdx = paa_size * i
        windowEndIdx = (paa_size * (i + 1)) - 1
        fullWindowStartIdx = math.ceil(windowStartIdx)
        fullWindowEndIdx = math.floor(windowEndIdx)
        startFraction = fullWindowStartIdx - windowStartIdx
        endFraction = windowEndIdx - fullWindowEndIdx
        if (startFraction > 0):
            fullWindowStartIdx = fullWindowStartIdx - 1
        if (endFraction > 0 and fullWindowEndIdx < window_size - 1):
            fullWindowEndIdx = fullWindowEndIdx + 1
        mapping.append([fullWindowStartIdx, fullWindowEndIdx + 1])
    return np.array(mapping)


class SaxdtMulti(object):
    def __init__(
            self,
            labels=None,
            random_state=None,
            custom_config=None,
            decision_tree_grid_search_params={
                'min_samples_split': [0.002, 0.01, 0.05, 0.1, 0.2],
                'min_samples_leaf': [0.001, 0.01, 0.05, 0.1, 0.2],
                'max_depth': [None, 2, 4, 6, 8, 10, 12, 16]
            },
            create_plotting_dictionaries=True
    ):
        self.labels = labels
        self.random_state = random_state
        self.decision_tree_grid_search_params = decision_tree_grid_search_params
        self.custom_config = custom_config

        self.X = None
        self.X_transformed = None
        self.y = None

        self.create_plotting_dictionaries = create_plotting_dictionaries
        self.subsequence_dictionary = None
        self.name_dictionary = None
        self.subsequences_norm_same_length = None

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        self.seql_models = list()
        X_transformed = list()
        feature_index = dict()
        prev_size = 0
        for dimension_idx in range(self.X.shape[2]):
            X = convert_numpy_to_sktime(self.X[:, :, dimension_idx: dimension_idx + 1])
            seql_model = MrSEQLClassifier(seql_mode='fs', symrep='sax', custom_config=self.custom_config)
            seql_model.fit(X, y)
            self.seql_models.append(seql_model)
            mr_seqs = seql_model._transform_time_series(X)
            X_transformed_dim = seql_model._to_feature_space(mr_seqs)
            X_transformed.append(X_transformed_dim)
            for i in range(np.array(X_transformed_dim).shape[1]):
                feature_index[i + prev_size] = [dimension_idx, i]
            prev_size += np.array(X_transformed_dim).shape[1]
        X_transformed = np.concatenate(X_transformed, axis=1)
        self.feature_index = feature_index

        clf = DecisionTreeClassifier()
        param_grid = self.decision_tree_grid_search_params
        grid = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', n_jobs=-1, verbose=0)
        grid.fit(X_transformed, y)

        clf = DecisionTreeClassifier(**grid.best_params_, random_state=self.random_state)
        clf.fit(X_transformed, y)
        prune_duplicate_leaves(clf)

        self.X_transformed = X_transformed
        self.decision_tree = clf
        self.decision_tree_explorable = NewTree(clf)
        self.decision_tree_explorable.build_tree()
        self._build_tree_graph()
        if self.create_plotting_dictionaries:
            self.create_dictionaries()
        return self

    def predict(self, X):
        X_transformed = list()
        for dimension_idx in range(X.shape[2]):
            X_dim = convert_numpy_to_sktime(X[:, :, dimension_idx: dimension_idx + 1])
            mr_seqs = self.seql_models[dimension_idx]._transform_time_series(X_dim)
            X_transformed.append(self.seql_models[dimension_idx]._to_feature_space(mr_seqs))
        X_transformed = np.concatenate(X_transformed, axis=1)
        y = self.decision_tree.predict(X_transformed)
        return y

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def _build_tree_graph(self, out_file=None):
        dot_data = tree.export_graphviz(self.decision_tree, out_file=out_file,
                                        class_names=self.labels,
                                        filled=True, rounded=True,
                                        special_characters=True)
        self.graph = graphviz.Source(dot_data)
        return self

    def create_dictionaries(self):
        self.subsequence_dictionary = create_subsequences_dictionary(
            self.X, self.X_transformed, self.X_transformed.shape[1], self.seql_models, self.feature_index
        )

    def plot_factual_and_counterfactual(
            self,
            x,
            x_label,
            **kwargs
    ):
        plot_factual_and_counterfactual2(self, x, x_label, **kwargs)
        return self
    #
    # def plot_binary_heatmap(self, x_label, **kwargs):
    #     plot_binary_heatmap(x_label, self.y, self.X_transformed, **kwargs)
    #     return self
    #
    def find_leaf_id(self, ts):
        ts_transformed = list()
        for dimension_idx in range(ts.shape[2]):
            ts_dim = convert_numpy_to_sktime(ts[:, :, dimension_idx: dimension_idx + 1])
            mr_seqs = self.seql_models[dimension_idx]._transform_time_series(ts_dim)
            ts_transformed.append(self.seql_models[dimension_idx]._to_feature_space(mr_seqs))
        ts_transformed = np.concatenate(ts_transformed, axis=1)
        leaf_id = self.decision_tree.apply(ts_transformed)[0]
        return leaf_id
    #
    # def coverage_score(self, leaf_id):
    #     return coverage_score_tree(self.decision_tree, leaf_id)
    #
    # def precision_score(self, leaf_id, y, X=None):
    #     if X is None:
    #         X = self.X_transformed.copy()
    #     else:
    #         X = convert_numpy_to_sktime(X)
    #         mr_seqs = self.seql_model._transform_time_series(X)
    #         X_transformed = self.seql_model._to_feature_space(mr_seqs)
    #         X = X_transformed
    #     return precision_score_tree(self.decision_tree, X, y, leaf_id)
    #
    # def create_dictionaries(self):
    #     (self.subsequence_dictionary,
    #      self.name_dictionary,
    #      self.subsequence_norm_dictionary) = create_subsequences_dictionary(
    #         self.X, self.X_transformed, self.X_transformed.shape[1], self.seql_model
    #     )
    #
    # def predict_explanation(self, x, x_label, **kwargs):
    #     return predict_explanation(self, x, x_label)
    #
    # def plot_subsequences_grid(self, n, m, starting_idx=0, random=False, color="mediumblue", **kwargs):
    #     if self.subsequence_dictionary is None:
    #         self.create_dictionaries()
    #     subsequence_list = list()
    #     for key in self.subsequence_dictionary:
    #         subsequence_list.append(self.subsequence_dictionary[key].mean(axis=0).ravel())
    #     plot_subsequences_grid(subsequence_list, n=n, m=m, starting_idx=starting_idx, random=random, color=color,
    #                            **kwargs)
    #     return self


def test_sax_multi():
    random_state = 0
    np.random.seed(0)
    dataset_name = "cbfmulti"

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_multivariate_cbf(random_state=random_state)

    clf = SaxdtMulti(labels=["cylinder", "bell", "funnel"], random_state=np.random.seed(0))
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


def test_rec_autoencoder():
    from late.datasets.datasets import build_multivariate_cbf
    import keras
    from variational_autoencoder import load_model
    from tests_multivariate_autoencoder import plot_mts_comparison

    random_state = 0
    np.random.seed(0)
    dataset_name = "cbfmulti"

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_multivariate_cbf(random_state=random_state)

    resnet = keras.models.load_model("./trained_models/cbfmulti/cbfmulti_resnet.h5")
    _, _, autoencoder = load_model("./trained_models/cbfmulti/cbfmulti_vae")

    encoder = autoencoder.layers[2]
    decoder = autoencoder.layers[3]

    i = 0
    plot_mts_comparison(X_train[i], decoder.predict(encoder.predict(X_train[i:i + 1]))[0, :, :])

    from late.blackboxes.blackbox_wrapper import BlackboxWrapper
    blackbox = BlackboxWrapper(resnet, 3, 2)
    from utils import reconstruction_accuracy
    print(reconstruction_accuracy(X_exp_test, encoder, decoder, blackbox))


if __name__ == "__main__":
    from late.datasets.datasets import build_multivariate_cbf
    import keras
    from variational_autoencoder import load_model
    from late.blackboxes.blackbox_wrapper import BlackboxWrapper
    from late.neighgen.neighborhood_generators import NeighborhoodGenerator
    from late.explainers.lasts import Lasts

    random_state = 0
    np.random.seed(0)
    dataset_name = "cbfmulti"

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_multivariate_cbf(random_state=random_state)

    resnet = keras.models.load_model("./trained_models/cbfmulti/cbfmulti_resnet.h5")
    blackbox = BlackboxWrapper(resnet, 3, 2)
    _, _, autoencoder = load_model("./trained_models/cbfmulti/cbfmulti_vae")

    encoder = autoencoder.layers[2]
    decoder = autoencoder.layers[3]

    i = 5
    x = X_exp_test[i: i + 1]
    neighborhood_generator = NeighborhoodGenerator(blackbox, decoder)
    neigh_kwargs = {
        "balance": False,
        "n": 500,
        "n_search": 10000,
        "threshold": 2,
        "sampling_kind": "uniform_sphere",
        "kind": "gaussian_matched",
        "vicinity_sampler_kwargs": {"distribution": np.random.normal, "distribution_kwargs": dict()},
        "verbose": True,
        "stopping_ratio": 0.01,
        "downward_only": True,
        "redo_search": True,
        "forced_balance_ratio": 0.5,
        "cut_radius": True
    }
    # neighborhood_generator = NormalGenerator()

    lasts_ = Lasts(blackbox,
                   encoder,
                   decoder,
                   x,
                   neighborhood_generator,
                   labels=["cylinder", "bell", "funnel"]
                   )

    out = lasts_.generate_neighborhood(**neigh_kwargs)
    # lasts_ = load("./saved/lasts_cbfmulti.joblib")

    plot_exemplars_and_counterexemplars(lasts_.Z_tilde, lasts_.y, lasts_.x, lasts_.z_tilde, lasts_.x_label,
                                        figsize=(10, 6))



    # surrogate = Sbgdt(shapelet_model_params={"max_iter": 50}, random_state=random_state)
    surrogate = SaxdtMulti(random_state=np.random.seed(20))

    # SHAPELET EXPLAINER
    lasts_.fit_surrogate(surrogate, binarize_labels=True)

    # lasts_.encoder = None
    # dump(lasts_, "lasts_cbfmulti.joblib")

    lasts_.plot_factual_and_counterfactual(figsize=(10, 6))


    clf = SaxdtMulti(labels=["cylinder", "bell", "funnel"], random_state=np.random.seed(0))
    clf.fit(X_train, y_train)
    i = 9
    clf.plot_factual_and_counterfactual(X_exp_test[i:i+1], y_exp_test[i], figsize=(10, 6))
