from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans


def prepare_DicisionTreeRegressor():
    model = DecisionTreeRegressor()
    params = dict()
    params['max_depth'] = [3, 6, 9]
    return model, params


def kmeans_builder(params):
    return KMeans(n_clusters=params[0], random_state=params[1])
