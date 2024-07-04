import time
import logging

import pandas as pd
from lore_explainer.datamanager import prepare_dataset
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from xailib.explainers.lime_explainer import LimeXAITabularExplainer
from xailib.explainers.lore_explainer import LegacyLoreTabularExplainer, LoreTabularExplainer
from xailib.models.sklearn_classifier_wrapper import sklearn_classifier_wrapper

from lore_sa.bbox import sklearn_classifier_bbox
from lore_sa.dataset import TabularDataset
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.neighgen import RandomGenerator
from lore_sa.neighgen.genetic import GeneticGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator


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
    preprocessor = ColumnTransformer(
       transformers=[
           ('num', StandardScaler(), [0,1,2,3,4,5]),
           ('cat', OrdinalEncoder(), [6,7,8,9,10,11,12,13])
        ]
    )
    model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))
    X_train, X_test, y_train, y_test = train_test_split(rdf.loc[:, 'age':'native-country'].values, rdf['class'].values,
                                                                test_size=0.3, random_state=42, stratify=rdf['class'].values)



    # bb = CatBoostClassifier(custom_loss=['Accuracy'], random_seed=42, logging_level='Silent')
    #bb = LogisticRegression(C=1, penalty='l2')
    logging.info("Training the classification model")
    model.fit(X_train, y_train)

    # Build a wrapper aroung the bbox
    bbox = sklearn_classifier_wrapper(model)

    inst = rdf[real_feature_names].values[18]
    logging.info("Selecting this instance to be classified: %s", inst)
    logging.info("Classified as %s", bbox.predict(inst.reshape(1, -1)))



    start = time.time()
    config = {'neigh_type' : 'random', 'size' : 1000 }
    logging.info("Configuring and initializing LORE Explainer with configuration %s", config)
    # # Create an explainer: LORE
    # explainer = LegacyLoreTabularExplainer(bbox)
    # # let the explainer to scan the training or test set
    # explainer.fit(df, class_field, config)
    #
    # logging.info("Building an explanation for the instance")
    # exp = explainer.explain(inst)
    # logging.info("The legacy explanation: \n%s", exp.exp)
    # end = time.time()
    # logging.info("Elapsed time to build the explanation: %s", end- start)

    explainer2 = LoreTabularExplainer(bbox)
    explainer2.fit(rdf, class_field, config)
    exp2 = explainer2.explain(inst)
    logging.info("The new explanation: \n%s", exp2.exp)


    start = time.time()

    # Use another explainer: LIME
    inst = df[feature_names].values[18]
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