import numpy as np
import scipy.io
from scipy.sparse.coo import coo_matrix
import matplotlib.pyplot as plt


import numpy as np
from numpy.core.numerictypes import _can_coerce_all
from numpy.lib.function_base import gradient

import scipy.io
from scipy.sparse.coo import coo_matrix

import matplotlib.pyplot as plt


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

    def update_user_features(self, t, learning_rate):
        for i in range(0, self.user_count):
            self.user_features[i, t] += learning_rate * \
                self.user_feature_gradient(user_row=i, wrt_user_idx=t)

    def update_item_features(self, t, learning_rate):
        for j in range(0, self.item_count):
            self.item_features[t, j] += learning_rate * \
                self.item_feature_gradient(item_col=j, wrt_item_idx=t)

    def train_model(self, learning_rate=0.1, iterations=1000):
        accu_array = []
        x_array = []
        R = self.data
        for i in range(iterations):
            for t in range(self.features):
                self.update_user_features(t, learning_rate=learning_rate)
                self.update_item_features(t, learning_rate=learning_rate)
                for a in range(self.item_count):
                    for b in range(self.user_count):
                        R[a, b] -= self.user_features[b, t] * \
                            self.item_features[t, a]

            if (i+1) % 50 == 0:
                mse = self.MSE()
                accu_array.append(mse)
                x_array.append(i+1)
                print(mse)

        plt.title("optimized SVD")
        plt.plot(x_array, accu_array)
        plt.xlabel('iteration')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()


def smallTest():
    d = np.array([[0, 2, 5], [0, 3, 0], [1, 0, 0]])

    d2 = SVDmf(d, 20)
    d2.train_model(learning_rate=0.1)


def fbDataTest():
    facebook_matrix = scipy.io.mmread("./facebook_combined.mtx")

    f_data = coo_matrix((4039, 4039), dtype=np.int8).toarray()

    fModel = SVDmf(f_data, 2)
    fModel.train_model(learning_rate=0.1, iterations=100)


smallTest()

# fbDataTest()
