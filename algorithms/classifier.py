import numpy as np
from sklearn.linear_model import LogisticRegression

def logModel(features, target):
    # features columns = return from a single feature
    # target = get_labels

    X = np.zeros((len(target),1))
    for i in features:
        i = np.asarray(i)
        X = np.hstack((X, np.transpose([i])))
    return LogisticRegression().fit(X, target)