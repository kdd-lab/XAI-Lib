#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:16:29 2020

@author: francesco
"""

from pyts.datasets import make_cylinder_bell_funnel, load_gunpoint, load_coffee
from tslearn.generators import random_walk_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler  # doctest: +NORMALIZE_WHITESPACE
import warnings
from tslearn.datasets import UCR_UEA_datasets


def load_ucr_dataset(name, verbose=True, exp_dataset_threshold=500, exp_dataset_ratio=.3, random_state=0):
    data_loader = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = data_loader.load_dataset(name)

    assert len(X_train.shape) == 3
    assert X_train.shape[1] != 1

    label_encoder = False
    if y_train.min() != 0:
        label_encoder = True
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        if label_encoder:
            print("Label Encoding:", le.classes_)
        print("X_train: ", X_train.shape)
        print("y_train: ", y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))
        print()
        print("X_test: ", X_test.shape)
        print("y_test: ", y_test.shape)
        unique, counts = np.unique(y_test, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_exp = X_train.copy()
    y_exp = y_train.copy()
    if X_train.shape[0] > exp_dataset_threshold:
        X_train, X_exp, y_train, y_exp = train_test_split(
            X_train,
            y_train,
            test_size=exp_dataset_ratio,
            stratify=y_train,
            random_state=random_state
        )
    else:
        if verbose:
            warnings.warn("Blackbox and Explanation sets are the same")

    return X_train, y_train, None, None, X_test, y_test, X_exp, y_exp, None, None, X_test, y_test


def build_cbf(n_samples=600, random_state=0, verbose=True):
    X_all, y_all = make_cylinder_bell_funnel(n_samples=n_samples, random_state=random_state)
    X_all = X_all[:, :, np.newaxis]

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all,
        y_all,
        test_size=0.3,
        stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp,
        y_exp,
        test_size=0.2,
        stratify=y_exp,
        random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train, y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_ts_syege(path="./datasets/ts_syege/", random_state=0, verbose=True):
    X_all = np.load(path + "ts_syege01.npy")
    X_all = X_all[:, :, np.newaxis]

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        # print("y SHAPE: ", y_all.shape)
        # unique, counts = np.unique(y_all, return_counts=True)
        # print("\nCLASSES BALANCE")
        # for i, label in enumerate(unique):
        # print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp = train_test_split(
        X_all,
        test_size=0.3,
        random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test = train_test_split(
        X_train,
        test_size=0.2,
        random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val = train_test_split(
        X_train,
        test_size=0.2,
        random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test = train_test_split(
        X_exp,
        test_size=0.2,
        random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val = train_test_split(
        X_exp_train,
        test_size=0.2,
        random_state=random_state
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, None, X_val, None, X_test, None, X_exp_train,
            None, X_exp_val, None, X_exp_test, None)


def build_multivariate_cbf(n_samples=600, n_features=3, random_state=0, verbose=True):
    X_all = [[], [], []]
    y_all = []
    for i in range(n_features):
        X, y = make_cylinder_bell_funnel(n_samples=n_samples, random_state=random_state + i)
        X = X[:, :, np.newaxis]
        for label in range(3):
            X_all[label].append(X[np.nonzero(y == label)])
    for i in range(len(X_all)):
        X_all[i] = np.concatenate(X_all[i], axis=2)
    for label in range(3):
        y_all.extend(label for i in range(len(X_all[label])))
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.array(y_all)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all,
        y_all,
        test_size=0.3,
        stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp,
        y_exp,
        test_size=0.2,
        stratify=y_exp,
        random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train, y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_rnd_blobs(
        n_ts_per_blob=200,
        sz=80,
        d=1,
        n_blobs=6,
        random_state=0,
        verbose=True
):
    X_all, y_all = random_walk_blobs(
        n_ts_per_blob=n_ts_per_blob,
        sz=sz,
        d=d,
        n_blobs=n_blobs,
        random_state=random_state
    )

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all,
        y_all,
        test_size=0.3,
        stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp,
        y_exp,
        test_size=0.2,
        stratify=y_exp,
        random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train, y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_synth(path="./datasets/synth03/", verbose=True, random_state=0):
    X_all = np.load(path + "X.npy")
    y_all = np.load(path + "y.npy").ravel()

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all,
        y_all,
        test_size=0.3,
        stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp,
        y_exp,
        test_size=0.2,
        stratify=y_exp,
        random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train, y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_har(path="./datasets/har_preprocessed/", verbose=True, label_encoder=True, random_state=0):
    X_all = np.load(path + "X_train.npy")
    y_all = np.load(path + "y_train.npy").ravel()
    if label_encoder:
        le = LabelEncoder()
        le.fit(y_all)
        y_all = le.transform(y_all)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_test = np.load(path + "X_test.npy")
    y_test = np.load(path + "y_test.npy").ravel()
    if label_encoder:
        y_test = le.transform(y_test)

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all,
        y_all,
        test_size=0.3,
        stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp,
        y_exp,
        test_size=0.2,
        stratify=y_exp,
        random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train, y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_fsdd(path="./datasets/fsdd/", verbose=True, label_encoder=True, random_state=0):
    df = pd.read_csv(path + "fsdd.csv", compression="gzip")

    X_all = df[df["indexes"] >= 5].drop(["indexes", "name"], axis=1).reset_index(drop=True)
    y_all = np.array(X_all["class"])
    X_all = X_all.drop("class", axis=1).values
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))

    X_test = df[df["indexes"] < 5].drop(["indexes", "name"], axis=1).reset_index(drop=True)
    y_test = np.array(X_test["class"])
    X_test = X_test.drop("class", axis=1).values
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    if verbose:
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all,
        y_all,
        test_size=0.4,
        stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp,
        y_exp,
        test_size=0.2,
        stratify=y_exp,
        random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train, y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train, y_exp_train,
            X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_esr(path="./datasets/esr/", verbose=True, random_state=0):
    X = pd.read_csv(path + "data.csv", index_col=0)
    y = np.array(X["y"])
    y_all = np.ravel(y).astype("int")
    for i in range(2, 6):
        y_all[y_all == i] = 2
    le = LabelEncoder()
    le.fit(y_all)
    y_all = le.transform(y_all)
    X_all = X.drop("y", axis=1).values
    rus = RandomUnderSampler(random_state=random_state, )
    X_all, y_all = rus.fit_resample(X_all, y_all)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))

    if verbose:
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                        test_size=0.2, stratify=y_all, random_state=random_state)

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(X_train, y_train,
                                                      test_size=0.3, stratify=y_train, random_state=random_state)

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, stratify=y_train, random_state=random_state)

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(X_exp, y_exp,
                                                                        test_size=0.2,
                                                                        stratify=y_exp,
                                                                        random_state=random_state)

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(X_exp_train, y_exp_train,
                                                                      test_size=0.2,
                                                                      stratify=y_exp_train,
                                                                      random_state=random_state)

    if verbose:
        print("SHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train, y_exp_train,
            X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_gunpoint(random_state=0, verbose=True, label_encoder=True):
    X_all, X_test, y_all, y_test = load_gunpoint(return_X_y=True)
    X_all = X_all[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]
    if label_encoder:
        le = LabelEncoder()
        le.fit(y_all)
        y_all = le.transform(y_all)
        y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        stratify=y_all, random_state=random_state
    )

    X_exp_train = X_train.copy()
    y_exp_train = y_train.copy()
    X_exp_val = X_val.copy()
    y_exp_val = y_val.copy()
    X_exp_test = X_test.copy()
    y_exp_test = y_test.copy()

    warnings.warn("Blackbox and Explanation sets are the same")

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)
        print("\nBlackbox and Explanation sets are the same!")

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_coffee(random_state=0, verbose=True, label_encoder=True):
    X_train, X_test, y_train, y_test = load_coffee(return_X_y=True)
    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]
    if label_encoder:
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_train.shape)
        print("y SHAPE: ", y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_exp_train = X_train.copy()
    y_exp_train = y_train.copy()
    X_exp_val = X_val = np.array(list())
    y_exp_val = y_val = np.array(list())
    X_exp_test = X_test.copy()
    y_exp_test = y_test.copy()

    warnings.warn("The validation sets are empty, use cross-validation to evaluate models")
    warnings.warn("Blackbox and Explanation sets are the same")

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_ecg200(path="./datasets/ECG200/", random_state=0, verbose=True, label_encoder=True):
    X_all = pd.read_csv(path + "ECG200_TRAIN.txt", sep="\s+", header=None)
    y_all = np.array(X_all[0])
    X_all = np.array(X_all.drop([0], axis=1))
    X_all = X_all[:, :, np.newaxis]

    X_test = pd.read_csv(path + "ECG200_TEST.txt", sep="\s+", header=None)
    y_test = np.array(X_test[0])
    X_test = np.array(X_test.drop([0], axis=1))
    X_test = X_test[:, :, np.newaxis]

    if label_encoder:
        le = LabelEncoder()
        le.fit(y_all)
        y_all = le.transform(y_all)
        y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        stratify=y_all, random_state=random_state
    )

    X_exp_train = X_train.copy()
    y_exp_train = y_train.copy()
    X_exp_val = X_val.copy()
    y_exp_val = y_val.copy()
    X_exp_test = X_test.copy()
    y_exp_test = y_test.copy()

    warnings.warn("Blackbox and Explanation sets are the same")

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


if __name__ == "__main__":
    (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
     y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test) = load_ucr_dataset("SmoothSubspace")

    import matplotlib.pyplot as plt
    from matplotlib import cm

    plt.plot(X_train[:10][:, :, 0].T)
    plt.show()
