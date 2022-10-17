from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans


def prepare_DicisionTreeRegressor():
    model = DecisionTreeRegressor()
    params = dict()
    params['max_depth'] = [3, 6, 9]
    return model, params


def kmeans_builder(params):
    return KMeans(n_clusters=params[0], random_state=params[1])


def kmeans_get_params():
    params = []
    # n_clusters
    params.append([2, 3, 4, 5, 6, 8])
    # random_state
    params.append([1, 10, 100, 1000])
    return params
