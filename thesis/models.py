import keras
from keras import layers
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from thesis import ModelBuilder
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor


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


class NeutralNetworkBuilder(ModelBuilder):
    DATA_NUMBER_OF_FEATURES = 13
    SECOND_LAYER_SIZE_INDEX = 0

    def get_model(self, params):
        size = params[self.SECOND_LAYER_SIZE_INDEX]

        def build_model():
            model = keras.Sequential([
                layers.Input(shape=(self.DATA_NUMBER_OF_FEATURES,)),
                layers.Dense(25, kernel_initializer='normal', activation='relu'),
                layers.Dense(1, kernel_initializer='normal')
            ])

            model.compile(loss='mean_squared_error', optimizer='adam')
            return model

        return KerasRegressor(build_fn=build_model, nb_epoch=100, batch_size=10)

    def get_param_lists(self):
        params = []
        # layers_size
        params.insert(self.SECOND_LAYER_SIZE_INDEX, [4, 8, 16])
        return super()._cluster_param_builder(params)
