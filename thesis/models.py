from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from thesis import ModelBuilder


class KMeansBuilder(ModelBuilder):
    N_CLUSTERS_INDEX = 0
    RANDOM_STATE_INDEX = 1

    def get_model(self, params):
        return KMeans(n_clusters=params[self.N_CLUSTERS_INDEX], random_state=params[self.RANDOM_STATE_INDEX])

    def get_param_lists(self):
        params = []
        # n_clusters
        params.insert(self.N_CLUSTERS_INDEX, [2, 3, 4, 5, 6, 8])
        # random_state
        params.insert(self.RANDOM_STATE_INDEX, [1, 10, 100, 1000])
        return super()._cluster_param_builder(params)


class DecisionTreeBuilder(ModelBuilder):
    MAX_DEPTH_INDEX = 0

    def get_model(self, params):
        return DecisionTreeRegressor(max_depth=params[self.MAX_DEPTH_INDEX])

    def get_param_lists(self):
        params = []
        # max_depth
        params.insert(self.MAX_DEPTH_INDEX, [3, 6, 9])
        return super()._cluster_param_builder(params)


def prepare_DicisionTreeRegressor():
    model = DecisionTreeRegressor()
    params = dict()
    params['max_depth'] = [3, 6, 9]
    return model, params



