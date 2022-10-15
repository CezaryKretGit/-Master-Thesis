from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans


def prepare_DicisionTreeRegressor():
    model = DecisionTreeRegressor()
    params = dict()
    params['max_depth'] = [3, 6, 9]
    return model, params
