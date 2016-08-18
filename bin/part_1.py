#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

logging.basicConfig(level=logging.DEBUG)

import pandas as pd



def data_etl(data_path):
    logging.info('ETL of data from path: %s' % data_path)

    # Load data
    data_df = pd.read_csv(data_path, sep='\t')
    data_df = data_df.fillna(data_df.mean())
    logging.info('Data description: \n%s' % data_df.describe())

    return data_df


def gen_x_y(data_df):
    # Create response, regressor label(s)
    response_label = 'target'
    regressor_labels = list(data_df._get_numeric_data().columns)
    regressor_labels.remove(response_label)
    logging.info('response_label: %s' % response_label)
    logging.info('regressor labels (%s) : %s' % (len(regressor_labels), regressor_labels))

    # Create X,y data sets
    X = data_df[regressor_labels].as_matrix()
    y = data_df[response_label].as_matrix()

    # Return
    return X, y


def pipeline(data_df):
    # Create datasets
    X, y = gen_x_y(data_df)

    # Instantiate objects for grid search
    pipe = Pipeline(steps=[('imputer', Imputer()), ('pca', PCA()), ('linear_regression', LinearRegression())])

    # Create grid parameters
    grid_options = dict(imputer__strategy=['mean', 'median', 'most_frequent'],
                        pca__n_components= [20, 50, 100, 150, 200],
                        linear_regression__normalize = [True, False])

    # Create grid search
    estimator = GridSearchCV(estimator=pipe, param_grid=grid_options, scoring=make_scorer(mean_squared_error), n_jobs=4)
    estimator.fit(X, y)

    print estimator.grid_scores_
    print estimator.best_estimator_
    print estimator.best_params_
    print estimator.best_score_

    for params, mean_score, scores in estimator.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))




# Functions
def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    data_df = data_etl('../data/input/codetest_train.txt')
    pipeline(data_df)


# Main section
if __name__ == '__main__':
    main()
