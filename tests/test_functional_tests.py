import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans

from sklearn.datasets import load_diabetes
import pandas as pd
from thesis import Pipeline


def prepare_data():
    X, y = load_diabetes(return_X_y=True)
    print(X)
    X = pd.DataFrame(data=X, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    print(X)
    X['labels'] = y
    print(X)
    return X.iloc[:300], X.iloc[300:]


pipeline = Pipeline(prepare_data,
                    [(DecisionTreeRegressor(), {'max_depth': [3, 6, 9]}), (
                        DecisionTreeRegressor(), {'max_depth': [4, 8]})],
                    [KMeans(n_clusters=5) ])

if __name__ == '__main__':
    print(sklearn.metrics.get_scorer_names())
    for result_reg, result_clu in pipeline.full_training():
        print("Best only reg: %f using %s" % (result_reg.best_score_, result_reg.best_params_))
        print("Best only clu: %f using %s" % (result_clu.best_score_, result_clu.best_params_))


