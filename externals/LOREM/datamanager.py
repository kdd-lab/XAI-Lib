import numpy as np
import pandas as pd

from collections import defaultdict

from scipy.io import arff
# from skmultilearn.dataset import load_from_arff


def prepare_dataset(df, class_name):

    df = remove_missing_values(df)

    numeric_columns = get_numeric_columns(df)

    rdf = df

    df, feature_names, class_values = one_hot_encoding(df, class_name)

    real_feature_names = get_real_feature_names(rdf, numeric_columns, class_name)

    rdf = rdf[real_feature_names + (class_values if isinstance(class_name, list) else [class_name])]

    features_map = get_features_map(feature_names, real_feature_names)

    return df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map


def get_features_map(feature_names, real_feature_names):
    features_map = defaultdict(dict)
    i = 0
    j = 0

    while i < len(feature_names) and j < len(real_feature_names):
        if feature_names[i] == real_feature_names[j]:
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
            j += 1
        elif feature_names[i].startswith(real_feature_names[j]):
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
        else:
            j += 1
    return features_map


def get_real_feature_names(rdf, numeric_columns, class_name):
    if isinstance(class_name, list):
        real_feature_names = [c for c in rdf.columns if c in numeric_columns and c not in class_name]
        real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c not in class_name]
    else:
        real_feature_names = [c for c in rdf.columns if c in numeric_columns and c != class_name]
        real_feature_names += [c for c in rdf.columns if c not in numeric_columns and c != class_name]
    return real_feature_names


def one_hot_encoding(df, class_name):
    if not isinstance(class_name, list):
        dfX = pd.get_dummies(df[[c for c in df.columns if c != class_name]], prefix_sep='=')
        class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
        dfY = df[class_name].map(class_name_map)
        df = pd.concat([dfX, dfY], axis=1, join_axes=[dfX.index])
        feature_names = list(dfX.columns)
        class_values = sorted(class_name_map)
    else: # isinstance(class_name, list)
        dfX = pd.get_dummies(df[[c for c in df.columns if c not in class_name]], prefix_sep='=')
        # class_name_map = {v: k for k, v in enumerate(sorted(class_name))}
        class_values = sorted(class_name)
        dfY = df[class_values]
        df = pd.concat([dfX, dfY], axis=1, join_axes=[dfX.index])
        feature_names = list(dfX.columns)
    return df, feature_names, class_values


def remove_missing_values(df):
    for column_name, nbr_missing in df.isna().sum().to_dict().items():
        if nbr_missing > 0:
            if column_name in df._get_numeric_data().columns:
                mean = df[column_name].mean()
                df[column_name].fillna(mean, inplace=True)
            else:
                mode = df[column_name].mode().values[0]
                df[column_name].fillna(mode, inplace=True)
    return df


def get_numeric_columns(df):
    numeric_columns = list(df._get_numeric_data().columns)
    return numeric_columns


def prepare_iris_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, skipinitialspace=True)
    return df, class_name


def prepare_wine_dataset(filename):
    class_name = 'quality'
    df = pd.read_csv(filename, skipinitialspace=True, sep=';')
    return df, class_name


def prepare_adult_dataset(filename):
    class_name = 'class'
    df = pd.read_csv(filename, skipinitialspace=True, na_values='?', keep_default_na=True)
    columns2remove = ['fnlwgt', 'education-num']
    df.drop(columns2remove, inplace=True, axis=1)
    return df, class_name


def prepare_german_dataset(filename):
    class_name = 'default'
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = [c.replace('=', '') for c in df.columns]
    return df, class_name


def prepare_compass_dataset(filename, binary=False):

    df = pd.read_csv(filename, delimiter=',', skipinitialspace=True)

    columns = ['age', 'age_cat', 'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    if binary:
        def get_class(x):
            if x < 7:
                return 'Medium-Low'
            else:
                return 'High'
        df['class'] = df['decile_score'].apply(get_class)
    else:
        df['class'] = df['score_text']

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    class_name = 'class'
    return df, class_name


def prepare_churn_dataset(filename):
    class_name = 'churn'
    df = pd.read_csv(filename, skipinitialspace=True, na_values='?', keep_default_na=True)
    columns2remove = ['phone number']
    df.drop(columns2remove, inplace=True, axis=1)
    return df, class_name


def prepare_yeast_dataset(filename):
    df = pd.DataFrame(arff.loadarff(filename)[0])

    for col in df.columns[-14:]:
        df[col] = df[col].apply(pd.to_numeric)

    cols_Y = [col for col in df.columns if col.startswith('Class')]
    # cols_X = [col for col in df.columns if col not in cols_Y]

    return df, cols_Y


def prepare_medical_dataset(filename):
    data = load_from_arff(filename, label_count=45, load_sparse=False, return_attribute_definitions=True)
    cols_X = [i[0] for i in data[2]]
    cols_Y = [i[0] for i in data[3]]
    X_med_df = pd.DataFrame(data[0].todense(), columns=cols_X)
    y_med_df = pd.DataFrame(data[1].todense(), columns=cols_Y)
    df = pd.concat([X_med_df, y_med_df], 1)

    return df, cols_Y


# https://www.kaggle.com/aniruddhachoudhury/credit-risk-model#train.csv/home/riccardo/Scaricati/bank.csv
def prepare_bank_dataset(filename):
    class_name = 'give_credit'
    df = pd.read_csv(filename, skipinitialspace=True, keep_default_na=True, index_col=0)
    return df, class_name


def prepare_fico_dataset(filename):
    class_name = 'RiskPerformance'
    df = pd.read_csv(filename, skipinitialspace=True, keep_default_na=True)
    return df, class_name

