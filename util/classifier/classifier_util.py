#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from util.res_process import SaveTime
from .classifier_base import classifier


def gen_label_info(data):
    unique, count = np.unique(data, return_counts=True)
    data_count = dict(zip(unique, count))
    return data_count


def classifier_method(train_X, train_y, test_X, test_y, selected_f, c_name, di=None):
    st = SaveTime()
    classifier_out = classifier(train_X[:, selected_f], train_y, test_X[:, selected_f], test_y, c_name)
    expense = st.get_spend_time()

    data_count = gen_label_info(test_y)
    if di is not None:
        di.update_item_classifier(c_name, classifier_out[0], classifier_out[1], expense, data_count)
        return di
    return classifier_out[0], classifier_out[1], expense
