from collections.abc import Callable
from models import regression
from models import clustering
import numpy as np
import pandas as pd

SCV_DATA_PATH = 'TO DO'


def prepare_data() -> (pd.DataFrame, pd.DataFrame):
    return pd.DataFrame(), pd.DataFrame()


def _ignore_cols(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return data.drop(columns=columns, inplace=False)


class Pipeline:

    def __init__(self, data_transform_function: Callable[[], (pd.DataFrame, pd.DataFrame)],
                 regression_model, cluster_model=None):
        self.prepare_data = data_transform_function
        self.regression_model = regression_model
        self.cluster_model = cluster_model

    def perform_full_experiment(self) -> (float, float):
        train_data, test_data = self.prepare_data()
        train_feats, train_labels = _ignore_cols(train_data, ['label']), train_data['label']
        test_feats, test_labels = _ignore_cols(test_data, ['label']), test_data['label']

        # clustering
        self.cluster_model.fit(_ignore_cols(train_data, ['label']))
        train_data['cluster'] = self.cluster_model.predict(train_feats)
        test_data['cluster'] = self.cluster_model.predict(test_feats)

        # regression learning not clustered
        self.regression_model.fit(train_feats, train_labels)
        only_regression_results = self.regression_model.score(test_feats, test_labels)

        # regression learning clustered data
        train_enriched_feats = _ignore_cols(train_data, ['label'])
        test_enriched_feats = _ignore_cols(test_data, ['label'])

        self.regression_model.fit(train_enriched_feats, train_labels)
        with_clustering_results = self.regression_model.score(test_enriched_feats, test_labels)

        return only_regression_results, with_clustering_results


if __name__ == '__main__':
    print(np.__version__)
    print(pd.__version__)
    exit()
