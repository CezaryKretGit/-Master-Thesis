
from sklearn.datasets import load_diabetes
import pandas as pd
from thesis import Pipeline
from thesis import KMeansBuilder, DecisionTreeBuilder


def prepare_testing_data():
    X, y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(data=X, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    X['labels'] = y
    return X.iloc[:300], X.iloc[300:]


def test_basic_pipeline():
    pipeline = Pipeline(prepare_testing_data,
                        [(DecisionTreeBuilder()), ],
                        [KMeansBuilder(), ])
    results = pipeline.full_training()
    print(results)
    print(results[0])
    results_df = pipeline.results_as_df(results)
    print(results_df)
    print(results_df.iloc[0])


if __name__ == '__main__':
    test_basic_pipeline()
