import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from lore_explainer.datamanager import prepare_dataset
from xailib.explainers.lime_explainer import LimeXAITabularExplainer
from xailib.explainers.lore_explainer import LoreTabularExplainer
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper
import time
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    source_file = 'datasets/adult.csv'
    class_field = 'class'
    # Load and transform dataset and select one row to classify and explain
    df = pd.read_csv(source_file, skipinitialspace=True, na_values='?', keep_default_na=True)
    df, feature_names, class_values, numeric_columns, \
    rdf, real_feature_names, features_map = prepare_dataset(df, class_field)
    print(df.head())
    print(class_values)
    # Learn a model from the data
    test_size = 0.3
    random_state = 0
    X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names].values, df[class_field].values,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=df[class_field].values)

    #bb = RandomForestClassifier(n_estimators=20, random_state=random_state)
    bb = CatBoostClassifier(custom_loss=['Accuracy'], random_seed=42, logging_level='Silent')
    #bb = LogisticRegression(C=1, penalty='l2')
    bb.fit(X_train, Y_train)

    # Build a wrapper aroung the bbox
    bbox = sklearn_classifier_wrapper(bb)

    inst = df[feature_names].values[18]
    print(inst, bbox.predict(inst.reshape(1, -1)))

    start = time.time()
    config = {'neigh_type' : 'geneticp', 'size' : 1000, 'ocr' : 0.1, 'ngen' : 10 }
    print("hello")
    print(config)
    # Create an explainer: LORE
    explainer = LoreTabularExplainer(bbox)
    # let the explainer to scan the training or test set
    explainer.fit(df, class_field, config)


    print('building an explanation')
    exp = explainer.explain(inst)
    print(exp)
    end = time.time()
    print('print ', end - start)

    start = time.time()
    print("hello")
    # Use another explainer: LIME
    config = {'feature_selection': 'lasso_path', 'discretize_continuous' : True, 'discretizer' : 'decile'}
    limeExplainer = LimeXAITabularExplainer(bbox)
    limeExplainer.fit(df, class_field, config)

    lime_exp = limeExplainer.explain(inst)
    print(lime_exp.as_list())
    end = time.time()
    print('print ', end - start)


