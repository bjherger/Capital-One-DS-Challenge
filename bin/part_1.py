#!/usr/bin/env python
"""
coding=utf-8

Code supporting Part 1 of Capital One Data Science challenge.

This section of the code challenge centers around creating a predictive model from an unknown / un-described data set,
with one response variable.

"""
import functools
import logging

import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

import pandas as pd

logging.basicConfig(level=logging.DEBUG)


def main():
    """
    Main code for Part 1 of C1 DS Challenge
    :return: None
    :rtype: None
    """
    # Create data set
    data_df = data_etl('../data/input/codetest_train.txt')

    # Grid search data set (results output to file as well)
    best_estimator = grid_search(data_df)
    # best_estimator = pickle.load(open('../data/output/part1/best_estimator.pkl', 'rb'))
    predict(best_estimator, '../data/input/codetest_test.txt')


def data_etl(data_path):
    """
    ETL the undescribed data
    :param data_path: Path to dataset
    :type data_path: str
    :return: ETL'd data
    :rtype: pd.DataFrame
    """
    logging.info('ETL of data from path: %s' % data_path)

    # Load data
    data_df = pd.read_csv(data_path, sep='\t')
    data_df = data_df.fillna(data_df.mean())
    logging.info('Data description: \n%s' % data_df.describe())
    logging.info('Data types: \n%s' % data_df.dtypes.value_counts())

    return data_df


def gen_x_y(data_df):
    """
    Generate X and y data for SKLearn consumption
    :param data_df: Un-described dataset
    :type data_df: pd.DataFrame
    :return: Tuple containing X and Y numpy matrices
    :rtype: (np.matrix, np.matrix)
    """

    # Create response, regressor label(s)
    response_label = 'target'
    number_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    regressor_labels = list(data_df.select_dtypes(include=number_types).columns)
    regressor_labels.remove(response_label)
    logging.info('response_label: %s' % response_label)
    logging.info('regressor labels (%s) : %s' % (len(regressor_labels), regressor_labels))

    # Create X,y data sets
    X = data_df[regressor_labels].as_matrix()
    y = data_df[response_label].as_matrix()

    # Return
    return X, y


def grid_search(data_df):
    """
    Run full grid search by creating one pipeline per ML algorithm, then running grid search on those pipelines.
    :param data_df: Un-described dataset
    :type data_df: pd.DataFrame
    :return: Best SKLearn estimator
    """
    # Create datasets
    X, y = gen_x_y(data_df)

    # OLS grid
    ols_dict = dict()
    ols_dict['model_type'] = 'ols'
    ols_dict['pipe'] = Pipeline(
        steps=[('imputer', Imputer()), ('pca', PCA()), ('linear_regression', LinearRegression())])
    ols_dict['param_grid'] = dict(imputer__strategy=['mean', 'median', 'most_frequent'],
                                  pca__n_components=[20, 50, 100, 150, 200, 250],
                                  linear_regression__normalize=[True, False])

    # RF grid
    rf_dict = dict()
    rf_dict['model_type'] = 'rf'
    rf_dict['pipe'] = Pipeline(
        steps=[('imputer', Imputer()), ('pca', PCA()), ('random_forest', RandomForestRegressor())])

    rf_dict['param_grid'] = dict(imputer__strategy=['mean', 'median', 'most_frequent'],
                                 pca__n_components=[20, 50, 100, 150, 200, 250],
                                 random_forest__n_estimators=[3, 5, 10, 20],
                                 random_forest__max_depth=[None, 5, 20, 50])

    # GBM grid
    gbm_dict = dict()
    gbm_dict['model_type'] = 'gbm'
    gbm_dict['pipe'] = Pipeline(steps=[('imputer', Imputer()), ('pca', PCA()), ('gbm', GradientBoostingRegressor())])

    gbm_dict['param_grid'] = dict(imputer__strategy=['mean', 'median', 'most_frequent'],
                                  pca__n_components=[20, 100, 200, 250],
                                  gbm__n_estimators=[25, 75],
                                  gbm__max_depth=[2, 8],
                                  gbm__random_state=[1])

    # Set up grid search
    model_list = [ols_dict, rf_dict, gbm_dict]
    # model_list = [ols_dict]
    model_df = pd.DataFrame(model_list)
    partialed_train_grid = functools.partial(train_grid, X=X, y=y)

    # Run grid search
    model_df['results'] = model_df.apply(func=partialed_train_grid, axis=1)

    # Unpack results
    results_list = model_df['results'].tolist()
    results_list = reduce(lambda x, y: x + y, results_list)
    results_df = pd.DataFrame(results_list)

    results_df[results_df['model_type'] == 'ols']['estimator'][0]

    # Output
    grid_results_output_path = '../data/output/part_part1_grid_results_total.csv'
    results_df.to_csv(grid_results_output_path)

    # Return best estimator (manually selected from reviewing results)
    best_estimator = results_df[results_df['model_type'] == 'ols']['estimator'][0].best_estimator_
    logging.info('Best estimator: %s' % best_estimator)
    pickle.dump(best_estimator, open('../data/output/part1/best_estimator.pkl', 'wb'))

    return best_estimator


def predict(best_estimator, test_data_path):
    # Read in test data
    test_df = data_etl(test_data_path)

    # Create dummy target column
    test_df['target'] = 0
    X, y = gen_x_y(test_df)

    test_pred = best_estimator.predict(X)
    test_pred_df = pd.DataFrame(test_pred)
    test_pred_df.columns = ['target_predicted']
    logging.info('Test predictions: %s' % test_pred_df)
    test_pred_df.to_csv('../data/output/part1/part1_test_predictions.csv', index=False, header=False)
    return test_df


def train_grid(model_dict, X, y):
    """
    Helper function to run grid search for a pipeline description on the given data
    :param model_dict: Dictionary contianing pipeline description, parameter grid
    :param X: X dataset to train SKLearn model on
    :type X: np.matrix
    :param y: Y response variable to train SKLearn model on
    :type y: np.matrix
    :return: Results of grid search
    :rtype: pd.DataFrame
    """
    # Unpack values
    grid_options = model_dict['param_grid']
    pipe = model_dict['pipe']

    model_type = model_dict['model_type']

    logging.info('Working on model type: %s' % model_type)

    # Create estimator
    estimator = GridSearchCV(estimator=pipe, param_grid=grid_options,
                             scoring=make_scorer(score_func=mean_squared_error, greater_is_better=False), n_jobs=-1)

    # Fit grid search
    estimator.fit(X, y)

    # Convert SKLearn grid search results to Pandas Data Frame
    grid_result_list = list()
    for params, mean_score, scores in estimator.grid_scores_:
        mean_score = -1 * mean_score
        line_dict = {'score': mean_score,
                     'score_std': scores.std(),
                     'score_ce_low': mean_score - scores.std(),
                     'score_ce_high': mean_score + scores.std(),
                     'params': params,
                     'estimator': estimator,
                     'model_type': model_type}

        line_dict.update(params)

        grid_result_list.append(line_dict)

    grid_result_df = pd.DataFrame(grid_result_list)

    # Output results
    grid_results_output_path = '../data/output/part1/part1_grid_results_' + model_type + '.csv'
    grid_result_df.to_csv(grid_results_output_path)
    logging.info('Results of %s model are in %s' % (model_type, grid_results_output_path))

    return grid_result_list


# Main section
if __name__ == '__main__':
    main()
