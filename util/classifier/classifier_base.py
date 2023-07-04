#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import warnings

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score

warnings.filterwarnings("ignore")


def get_classifier(c_name):
    """
    根据分类器名称 返回分类器函数
    :param c_name:
    :return: 分类器函数
    """
    classifier_list = ["svm", "knn", 'gnb', "decision tree", "RandomForest", "LogisticRegression"]
    classifier_list1 = ["svm", "knn", 'gnb', "dt", "rf", "lr", "mlp"]
    if c_name == 'svm':
        classifier = SVC(gamma='auto')
    elif c_name == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    elif c_name == 'gnb':
        classifier = GaussianNB()
    elif c_name == 'bdnb':
        classifier = MultinomialNB()
    elif c_name == 'dt':
        classifier = DecisionTreeClassifier()
    elif c_name == 'rf':
        classifier = RandomForestClassifier(n_estimators=100)
    elif c_name == 'lr':
        classifier = LogisticRegression()
    elif c_name == 'mlp':
        classifier = MLPClassifier()
    else:
        raise Exception("{} classifier not existence".format(c_name))
    return classifier


def classifier(train_X, train_y, test_X, test_y, c_name):
    cls = get_classifier(c_name)
    cls.fit(train_X, train_y)
    y_pred = cls.predict(test_X)
    return accuracy_score(test_y, y_pred), f1_score(test_y, y_pred, average='macro', labels=np.unique(y_pred))
