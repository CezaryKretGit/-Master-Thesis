from collections.abc import Callable
from abc import abstractmethod
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
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


class ModelBuilder:

    @abstractmethod
    def get_model(self, params):
        pass

    @abstractmethod
    def get_param_lists(self):
        pass

    def _cluster_param_builder(self, params_list: list):
        params = Param()
        param = params_list.pop(0)

        for param_value in param:
            self._param_builder(params_list.copy(), [param_value, ], params)

        return params.super_list

    def _param_builder(self, params_list: list, element_list: list, super_list):
        if len(params_list) == 0:
            super_list.append(element_list)
            return

        param = params_list.pop(0)

        for param_value in param:
            self._param_builder(params_list.copy(), element_list + [param_value, ], super_list)


class Pipeline:

    def __init__(self, data_transform_function: Callable[[], (pd.DataFrame, pd.DataFrame)],
                 regression_models: list[ModelBuilder], cluster_models: list[ModelBuilder]):
        self.regression_models_with_params = regression_models
        self.cluster_models_with_params = cluster_models
        self.train_data, self.test_data = data_transform_function()
        self.results = None

    def full_training(self):

        results = []
        # for each cluster model
        for cluster_builder in self.cluster_models_with_params:
            # for each regression model
            for regression_builder in self.regression_models_with_params:
                # for each cluster parameters
                for cluster_params in cluster_builder.get_param_lists():
                    # for each regression parameters
                    for regression_params in regression_builder.get_param_lists():
                        # do an experiment
                        results.append(list(self.experiment(cluster_builder.get_model(cluster_params),
                                                            regression_builder.get_model(regression_params))) +
                                       [cluster_params, regression_params])

        return results

    def experiment(self, cluster_model, regression_model, train_data=None):
        if train_data is None:
            train_data = self.train_data
        train_feats, train_labels = _ignore_cols(train_data, ['labels', 'cluster']), train_data['labels']

        # clustering
        cluster_model.fit(train_feats)
        train_data['cluster'] = cluster_model.predict(train_feats)

        # regression learning not clustered
        only_regression_results = cross_validate(estimator=regression_model, X=train_feats, y=train_labels,
                                                 cv=5, scoring='r2')

        # regression learning clustered data
        train_enriched_feats = _ignore_cols(train_data, ['labels'])
        with_clustering_results = cross_validate(estimator=regression_model, X=train_enriched_feats, y=train_labels,
                                                 cv=5, scoring='r2')

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
