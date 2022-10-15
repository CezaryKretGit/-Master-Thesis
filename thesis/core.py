from collections.abc import Callable
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

SCV_DATA_PATH = 'TO DO'


def example_prepare_data() -> (pd.DataFrame, pd.DataFrame):
    return pd.DataFrame(), pd.DataFrame()


def _ignore_cols(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return data.drop(columns=columns, inplace=False, errors='ignore')


class Param:
    def __init__(self):
        self.super_list = []

    def append(self, val):
        self.super_list.append(val)


def cluster_param_builder(params_list: list):
    params = Param()

    param = params_list.pop(0)

    for param_value in param:
        _param_builder(params_list.copy(), [param_value, ], params)

    return params.super_list


def _param_builder(params_list: list, element_list: list, super_list):
    if len(params_list) == 0:
        super_list.append(element_list)
        return

    param = params_list.pop(0)

    for param_value in param:
        _param_builder(params_list.copy(), element_list + [param_value, ], super_list)


class Pipeline:

    def __init__(self, data_transform_function: Callable[[], (pd.DataFrame, pd.DataFrame)],
                 regression_models: list, cluster_models: list):
        self.regression_models_with_params = regression_models
        self.cluster_models_with_params = cluster_models
        self.train_data, self.test_data = data_transform_function()
        self.results = None

    def full_training(self):

        results = []
        print(self.cluster_models_with_params)
        for cluster_builder, cluster_all_params in self.cluster_models_with_params:
            for cluster_params in cluster_param_builder(cluster_all_params):
                print(cluster_params)
                for regression_model, params in self.regression_models_with_params:
                    results.append(list(self.experiment(cluster_builder(cluster_params), regression_model, params)) +
                                   [cluster_params,])

        return results

    def experiment(self, cluster_model, regression_model, regression_vals, train_data=None):
        if train_data is None:
            train_data = self.train_data
        train_feats, train_labels = _ignore_cols(train_data, ['labels', 'cluster']), train_data['labels']

        # clustering
        cluster_model.fit(train_feats)
        train_data['cluster'] = cluster_model.predict(train_feats)

        # regression learning not clustered
        grid_search = GridSearchCV(estimator=regression_model, param_grid=regression_vals, scoring='r2',
                                   cv=6, refit=True, return_train_score=True)
        only_regression_results = grid_search.fit(train_feats, train_labels)

        # regression learning clustered data
        train_enriched_feats = _ignore_cols(train_data, ['labels'])

        grid_search = GridSearchCV(estimator=regression_model, param_grid=regression_vals, scoring='r2',
                                   cv=6, refit=True, return_train_score=True)
        with_clustering_results = grid_search.fit(train_enriched_feats, train_labels)

        return only_regression_results, with_clustering_results

    def results_as_df(self, results=None):
        if results is None:
            return self.results

        data = [[result_reg.estimator,
                 result_reg.best_score_, result_reg.best_params_,
                 result_clu.best_score_, result_clu.best_params_, cluster_params]
                for result_reg, result_clu, cluster_params in results]

        columns = ['model', 'regression_only_score', 'regression_only_params',
                   'with_clustering_score', 'with_clustering_params', 'cluster_params']
        return pd.DataFrame(data=data, columns=columns)


if __name__ == '__main__':
    exit()
