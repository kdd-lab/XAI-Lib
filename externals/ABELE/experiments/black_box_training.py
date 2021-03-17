import sys

import json

from sklearn.metrics import accuracy_score, classification_report

from experiments.exputil import get_dataset
from experiments.exputil import train_black_box
from experiments.exputil import get_black_box


import warnings
warnings.filterwarnings('ignore')


def main():

    random_state = 0
    dataset = 'fashion'
    black_box = 'RF'
    print(dataset, black_box)

    path = './'
    path_models = path + 'models/'
    path_results = path + 'results/bb/'

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)
    results_filename = path_results + '%s_%s.json' % (dataset, black_box)

    X_train, Y_train, X_test, Y_test, use_rgb = get_dataset(dataset)
    train_black_box(X_train, Y_train, dataset, black_box, black_box_filename, use_rgb, random_state)
    bb_predict, bb_predict_proba = get_black_box(black_box, black_box_filename, use_rgb)

    Y_pred = bb_predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    cr = classification_report(Y_test, Y_pred)
    print('Accuracy: %.2f' % acc)
    print('Classification Report')
    print(cr)
    cr = classification_report(Y_test, Y_pred, output_dict=True)
    res = {
        'dataset': dataset,
        'black_box': black_box,
        'accuracy': acc,
        'report': cr
    }
    results = open(results_filename, 'w')
    results.write('%s\n' % json.dumps(res))
    results.close()


if __name__ == "__main__":
    main()
