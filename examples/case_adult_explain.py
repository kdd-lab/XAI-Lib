import time
import logging

import pandas as pd
from lore_explainer.datamanager import prepare_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xailib.explainers.lime_explainer import LimeXAITabularExplainer
from xailib.explainers.lore_explainer import LegacyLoreTabularExplainer, LoreTabularExplainer
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper


logging.basicConfig(level=logging.INFO)

def main():
    source_file = '../datasets/adult.csv'
    class_field = 'class'
    # Load and transform dataset and select one row to classify and explain
    df = pd.read_csv(source_file, skipinitialspace=True, na_values='?', keep_default_na=True)
    df, feature_names, class_values, numeric_columns, \
    rdf, real_feature_names, features_map = prepare_dataset(df, class_field)
    logging.info("Completed loading of dataset with size %s", df.shape)
    logging.info("Class values: %s", class_values)
    # Learn a model from the data
    test_size = 0.3
    random_state = 0
    X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names].values, df[class_field].values,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=df[class_field].values)

    bb = RandomForestClassifier(n_estimators=20, random_state=random_state)
    # bb = CatBoostClassifier(custom_loss=['Accuracy'], random_seed=42, logging_level='Silent')
    #bb = LogisticRegression(C=1, penalty='l2')
    logging.info("Training the classification model")
    bb.fit(X_train, Y_train)

    # Build a wrapper aroung the bbox
    bbox = sklearn_classifier_wrapper(bb)

    inst = df[feature_names].values[18]
    logging.info("Selecting this instance to be classified: %s", inst)
    logging.info("Classified as %s", bbox.predict(inst.reshape(1, -1)))



    start = time.time()
    config = {'neigh_type' : 'geneticp', 'size' : 1000, 'ocr' : 0.1, 'ngen' : 10 }
    logging.info("Configuring and initializing LORE Explainer with configuration %s", config)
    # Create an explainer: LORE
    explainer = LegacyLoreTabularExplainer(bbox)
    # let the explainer to scan the training or test set
    explainer.fit(df, class_field, config)

    logging.info("Building an explanation for the instance")
    exp = explainer.explain(inst)
    logging.info("The legacy explanation: \n%s", exp.exp)
    end = time.time()
    logging.info("Elapsed time to build the explanation: %s", end- start)

    explainer2 = LoreTabularExplainer(bbox)
    explainer2.fit(df, class_field, config)
    exp2 = explainer2.explain(inst)
    logging.info("The new explanation: \n%s", exp2.exp)


    start = time.time()

    # Use another explainer: LIME
    config = {'feature_selection': 'lasso_path', 'discretize_continuous' : True, 'discretizer' : 'decile'}
    logging.info("Configuring and initializing LIME explainer, with config: \n%s", config)
    limeExplainer = LimeXAITabularExplainer(bbox)
    limeExplainer.fit(df, class_field, config)

    lime_exp = limeExplainer.explain(inst)
    logging.info("LIME feature relevance list:\n%s", lime_exp.exp.as_list())
    end = time.time()
    logging.info("Elapsed time to build the explanation: %s", end - start)


if __name__ == '__main__':
    main()