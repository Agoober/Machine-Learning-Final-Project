import numpy as np
from numpy.core.numerictypes import _can_coerce_all
from numpy.lib.function_base import gradient

import scipy.io
from scipy.sparse.coo import coo_matrix

import matplotlib.pyplot as plt

# Part of this code is refering the tutorial from
# https://towardsdatascience.com/singular-value-decomposition-example-in-python-dab2507d85a0
# https://towardsdatascience.com/recommender-systems-in-python-from-scratch-643c8fc4f704


class SVDmf():

    def __init__(self, data, features):
        import numpy as np

        self.data = data
        self.features = features
        self.user_count = data.shape[0]
        self.item_count = data.shape[1]
        self.user_features = np.random.uniform(
            low=0.1, high=0.9, size=(self.user_count, self.features))
        self.item_features = np.random.uniform(
            low=0.1, high=0.9, size=(self.features, self.item_count))

    def MSE(self):
        matrix_product = np.matmul(self.user_features, self.item_features)
        return np.sum((self.data - matrix_product)**2)

    def single_gradient(self, user_row, item_col, wrt_user_idx=None, wrt_item_idx=None):

        u_row = self.user_features[user_row, :]
        i_col = self.item_features[:, item_col]

        ui_rating = float(self.data[user_row, item_col])
        prediction = float(np.dot(u_row, i_col))

        if wrt_user_idx != None:
            row_elem = float(i_col[wrt_user_idx])
            gradient = 2 * (ui_rating - prediction) * row_elem
        else:
            col_elem = float(u_row[wrt_item_idx])
            gradient = 2 * (ui_rating - prediction)*col_elem
        return gradient

    def user_feature_gradient(self, user_row, wrt_user_idx):
        sum_g = 0
        for col in range(0, self.item_count):
            sum_g += self.single_gradient(user_row=user_row,
                                          item_col=col, wrt_user_idx=wrt_user_idx)

        return sum_g/self.item_count

    def item_feature_gradient(self, item_col, wrt_item_idx):
        sum_g = 0
        for row in range(0, self.user_count):
            sum_g += self.single_gradient(user_row=row,
                                          item_col=item_col, wrt_item_idx=wrt_item_idx)
        return sum_g/self.item_count

    def update_user_features(self, learning_rate):
        for i in range(0, self.user_count):
            for j in range(0, self.features):
                self.user_features[i, j] += learning_rate * \
                    self.user_feature_gradient(user_row=i, wrt_user_idx=j)

    def update_item_features(self, learning_rate):
        for i in range(0, self.features):
            for j in range(0, self.item_count):
                self.item_features[i, j] += learning_rate * \
                    self.item_feature_gradient(item_col=j, wrt_item_idx=i)

    def train_model(self, learning_rate=0.1, iterations=1000):
        accu_array = []
        x_array = []
        for i in range(iterations):
            self.update_user_features(learning_rate=learning_rate)
            self.update_item_features(learning_rate=learning_rate)
            if (i+1) % 50 == 0:
                mse = self.MSE()
                accu_array.append(mse)
                x_array.append(i+1)
                print(mse)

        plt.title("SVD")
        plt.plot(x_array, accu_array)
        plt.xlabel('iteration')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()


def smallTest():
    d = np.array([[0, 2, 5], [0, 3, 0], [1, 0, 0]])

    d2 = SVDmf(d, 80)
    d2.train_model(learning_rate=0.1, iterations=1000)


def fbDataTest():
    facebook_matrix = scipy.io.mmread("./facebook_combined.mtx")

    f_data = coo_matrix((4039, 4039), dtype=np.int8).toarray()

    fModel = SVDmf(f_data, 3)
    fModel.train_model(learning_rate=0.1)


smallTest()

# fbDataTest()
