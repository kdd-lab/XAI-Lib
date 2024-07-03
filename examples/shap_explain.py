import pandas as pd
import shap
import numpy as np
import time
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from externals.LORE.datamanager import prepare_dataset
from xailib.explainers.shap_explainer_tab import ShapXAITabularExplainer
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
from xailib.models.keras_classifier_wrapper import keras_classifier_wrapper
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

if __name__ == '__main__':
    source_file = '../datasets/german_credit.csv'
    class_field = 'default'
    # Load and transform dataset and select one row to classify and explain
    df = pd.read_csv(source_file, skipinitialspace=True, na_values='?', keep_default_na=True)
    df, feature_names, class_values, numeric_columns, \
    rdf, real_feature_names, features_map = prepare_dataset(df, class_field)
    print(df.head())
    print(class_values)
    # Learn a model from the data
    test_size = 0.3
    random_state = 0
    X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names], df[class_field],
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=df[class_field])
    #bb = LogisticRegression(C=1, penalty='l2')
    #bb = CatBoostClassifier(custom_loss=['Accuracy'],random_seed=42,logging_level='Silent')
    bb = RandomForestClassifier(n_estimators=20, random_state=random_state)
    bb.fit(X_train.values, Y_train.values)
    bbox = sklearn_classifier_wrapper(bb)
    inst = X_train.iloc[18].values
    print(inst, bb.predict(inst.reshape(1, -1)))
    start = time.time()
    print("hello")
    explainer = ShapXAITabularExplainer(bbox)
    print(X_train.shape)
    #background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    config = {'explainer' : 'tree', 'X_train' : X_train.iloc[0:100].values, 'feature_pert' : 'interventional'}
    explainer.fit(config)


    print('building an explanation')
    exp = explainer.explain(inst)
    print(exp)
    end = time.time()
    print('time ', end - start)
    #explainer.force_plot(inst, 18)


