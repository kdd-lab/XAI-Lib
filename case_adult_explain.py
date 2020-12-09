import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from lore.datamanager import prepare_dataset
from xailib.explainers.lime_explainer import LimeXAITabularExplainer
from xailib.explainers.lore_explainer import LoreTabularExplainer
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper


if __name__ == '__main__':
    source_file = 'datasets/adult.csv'
    class_field = 'class'

    # Load and transform dataset and select one row to classify and explain
    df = pd.read_csv(source_file, skipinitialspace=True, na_values='?', keep_default_na=True)
    df, feature_names, class_values, numeric_columns, \
    rdf, real_feature_names, features_map = prepare_dataset(df, class_field)

    # Learn a model from the data
    test_size = 0.3
    random_state = 0
    X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names].values, df[class_field].values,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=df[class_field].values)

    bb = RandomForestClassifier(n_estimators=20, random_state=random_state)
    bb.fit(X_train, Y_train)

    # Build a wrapper aroung the bbox
    bbox = sklearn_classifier_wrapper(bb)

    inst = df[feature_names].values[18]
    print(inst, bbox.predict(inst.reshape(1, -1)))

    # Create an explainer: LORE
    explainer = LoreTabularExplainer(bbox)
    # let the explainer to scan the training or test set
    explainer.fit(df, class_field)

    print('building an explanation')
    exp = explainer.explain(inst)
    print(exp)

    # Use another explainer: LIME
    limeExplainer = LimeXAITabularExplainer(bbox)
    limeExplainer.fit(df, class_field)

    lime_exp = limeExplainer.explain(inst)
    print(lime_exp.as_list())
