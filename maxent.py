import numpy as np


def find_feature_expectations(demonstrations, feature_expectations, env):

    for episode in demonstrations:
        for i in range(6):
            feature_expectations += env.feature_vector(episode[0: i+1])

    feature_expectations /= len(demonstrations)


def irl(expert, learner, alpha, lr):
    gradient = expert - learner
    alpha += lr * gradient
    return gradient
