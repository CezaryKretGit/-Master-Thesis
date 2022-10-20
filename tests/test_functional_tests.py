from pandas import read_csv
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from thesis import Pipeline, NeutralNetworkBuilder
from thesis import KMeansBuilder, DecisionTreeBuilder
from sklearn import preprocessing

pd.set_option('display.max_columns', None)


def prepare_testing_data():
    x, y = load_diabetes(return_X_y=True)
    x = preprocessing.normalize(x)
    print(x)
    y = y / max(y)
    x = pd.DataFrame(data=x, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    x['labels'] = y
    print(x.head())
    return x.iloc[:300], x.iloc[300:]


def prepare_testing_data_2():
    dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:13]
    Y = dataset[:, 13]
    return X, Y


def test_basic_pipeline():
    pipeline = Pipeline(prepare_testing_data,
                        [(DecisionTreeBuilder()), ],
                        [KMeansBuilder(), ])
    results = pipeline.full_training()
    print(results)
    print(results[0])
    results_df = pipeline.save_results_as_df(results)
    print(results_df)
    print(results_df.iloc[0])


def test_nn_pipeline():
    pipeline = Pipeline(prepare_testing_data,
                        [NeutralNetworkBuilder(), ],
                        [KMeansBuilder(), ])
    results = pipeline.full_training()
    print(results)
    print(results[0])
    results_df = pipeline.save_results_as_df(results)
    print(results_df)
    print(results_df.iloc[0])


def test_nn_model():
    X, Y = prepare_testing_data_2()

    nn_builder = NeutralNetworkBuilder()
    model = nn_builder.get_model(nn_builder.get_param_lists()[0])
    ss = StandardScaler()
    ss.fit(X)
    X_trans = ss.transform(X)
    results = cross_val_score(model, X, Y, cv=10, scoring='neg_mean_squared_error')
    print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))


if __name__ == '__main__':
    test_nn_model()
