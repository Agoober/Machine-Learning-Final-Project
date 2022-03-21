#!/usr/bin/env python3

import numpy as np
from numpy import random

import scipy.io
from scipy.sparse.coo import coo_matrix
import matplotlib.pyplot as plt


class randomMF():

    def __init__(self, data, features):
        self.data = data
        self.features = features
        self.user_count = data.shape[0]
        self.item_count = data.shape[1]

        self.user_features = np.zeros([self.user_count, self.features])
        self.item_features = np.zeros([self.features, self.item_count])

    def MSE(self):
        matrix_product = np.matmul(self.user_features, self.item_features)
        return np.sum((self.data - matrix_product)**2)

    def randomUpdate(self):
        for user_idx in range(self.user_count):
            for usrFea_idx in range(self.features):
                self.user_features[user_idx, usrFea_idx] = random.rand()

            for item_idx in range(self.item_count):
                for itmFea_idx in range(self.features):
                    if itmFea_idx == self.features - 1:
                        self.item_features[itmFea_idx, item_idx] = self.data[user_idx,
                                                                             item_idx] / np.sum(self.user_features[user_idx])
                    else:
                        self.item_features[itmFea_idx,
                                           item_idx] = random.rand()

    def train_model(self, iterations=1000):
        accu_array = []
        x_array = []
        for i in range(iterations):
            self.randomUpdate()

            if (i+1) % 50 == 0:
                mse = self.MSE()
                accu_array.append(mse)
                x_array.append(i+1)
                print(mse)

        plt.title("random_approach")
        plt.plot(x_array, accu_array)
        plt.xlabel('iterations')
        plt.ylabel('MSE')
        plt.show()


def smallTest():
    d = np.array([[0, 2, 5], [0, 3, 0], [1, 0, 0]])

    d2 = randomMF(d, 5)
    d2.train_model(iterations=1000)


def fbDataTest():
    facebook_matrix = scipy.io.mmread("./facebook_combined.mtx")

    f_data = coo_matrix((4039, 4039), dtype=np.int8).toarray()

    fModel = randomMF(f_data, 3)
    fModel.train_model()


smallTest()

# fbDataTest()
