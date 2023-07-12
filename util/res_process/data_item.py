#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime


class DataItem:
    def __init__(self, reduce_stat=False):
        self.reduce_stat = reduce_stat

        self.data_item = {}

        self.item_init()

    def item_init(self):
        dataset_item = {"now_time": 0,
                        "dataset_name": "",
                        "original_feature_num": 0,
                        "original_samples_num": 0,
                        "division_rate": 0, }
        reduce_item = {"reduce_dim_name": '',
                       "reduce_dim_output": '',
                       "reduce_dim_expense": 0, }
        fs_item = {"feature_select_name": "",
                   "feature_select_output": "",
                   "feature_select_expense": 0, }
        classifier_item = {"classifier_name": '',
                           "classify_acc": 0,
                           "classify_recall": 0,
                           "classify_expense": 0,
                           "positive_negative_samples_num": 0,
                           }
        self.data_item.update(dataset_item)
        if self.reduce_stat:
            self.data_item.update(reduce_item)
        self.data_item.update(fs_item)
        self.data_item.update(classifier_item)

    def item_context_add(self, name, context):
        self.data_item[name] = self.data_item[name] + str(context)

    def item_context_change(self, name, context):
        self.data_item[name] = str(context)

    def update_time(self):
        self.data_item["now_time"] = datetime.now()

    def update_item_origin(self, dataset_name, original_feature_num, original_samples_num, division_rate):
        self.data_item["dataset_name"] = dataset_name
        self.data_item["original_feature_num"] = original_feature_num
        self.data_item["original_samples_num"] = original_samples_num
        self.data_item["division_rate"] = division_rate

    def update_item_reduce_dim(self, reduce_dim_name, reduce_dim_output, reduce_dim_expense):
        self.data_item["reduce_dim_name"] = reduce_dim_name
        self.data_item["reduce_dim_output"] = reduce_dim_output
        self.data_item["reduce_dim_expense"] = reduce_dim_expense

    def update_item_fs(self, feature_select_name, feature_select_output, feature_select_expense):
        self.data_item["feature_select_name"] = feature_select_name
        self.data_item["feature_select_output"] = feature_select_output
        self.data_item["feature_select_expense"] = feature_select_expense

    def update_item_classifier(self, classifier_name, classify_acc, classify_recall, classify_expense,
                               positive_negative_samples_num):
        self.data_item["classifier_name"] = classifier_name
        self.data_item["classify_acc"] = classify_acc
        self.data_item["classify_recall"] = classify_recall
        self.data_item["classify_expense"] = classify_expense
        self.data_item["positive_negative_samples_num"] = positive_negative_samples_num

    def get_acc(self):
        return self.data_item['classify_acc']

    def get_value(self):
        return self.data_item.values()

    def get_feature_select_expense(self):
        return self.data_item['feature_select_expense']
