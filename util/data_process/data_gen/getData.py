#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import random
import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold

from util.decorate import working_dir


def load_dataset2npy(dataset_name, del_comma=True, save_flag=False):
    """
    将其他文本的数据集转为npy  可选保存  返回nparray
    :param dataset_name:
    :param del_comma: 删除每行末尾逗号
    :param save_flag:
    :return:
    """
    f = open("./dataSet/" + dataset_name, 'r', encoding='UTF-8-sig')  # 因为有bom所以用sig
    lines = f.readlines()
    data = list()

    for line in lines[:]:
        info = line.split(",")
        if del_comma:
            info[-1] = info[-1][:-1]
        data.append(info)
    f.close()

    a = np.array(data)
    # a = np.delete(a, 31, axis=0)
    # a = a.astype(np.float64)
    if save_flag:
        # print("./dataSet/" + dataset_name.split('.')[0] + "_origin.npy")
        np.save("./dataSet/" + dataset_name.split('.')[0] + "_origin.npy", a)  # 保存为.npy格式
    return a


def get_element(data):
    """
    每一列有什么元素
    :param data:
    :return:
    """
    i = 0  # 第几列
    for col in data.T:
        c = np.unique(col)
        i = i + 1
        print(i, len(c))
        print(c)


def load_mat(path):
    """
    把mat以npy格式返回
    :param path:
    :return:
    """
    mat = sio.loadmat(path)
    X = mat['X']  # data
    X = X.astype(float)
    y = mat['Y']  # label
    y = np.array([y[:, 0]]).T
    data = np.hstack((X, y))
    return data


@working_dir()
def get_data(file_name, absolute_path=False):
    """
    根据文件名去不同文件夹加载数据
    :param absolute_path: 是否启动绝对路径  不启用的话会在dataset路径下检索
    :param file_name:
    :return:
    """
    if absolute_path:
        data = np.load(file_name)
    else:
        base_path = "./dataSet/"
        if file_name.find("mat") != -1:
            data = load_mat(base_path + "mat/" + file_name)
        elif file_name.find("dis") != -1:
            data = np.load(base_path + "discrete/" + file_name, allow_pickle=True)
        elif file_name.find("origin") != -1:
            data = np.load(base_path + "origin/" + file_name, allow_pickle=True)
        else:
            data = np.load(base_path + "normal/" + file_name, allow_pickle=True)
    return data


def load_data(file_name, di=None, division_rate=-1, samples_num=-1):
    """
    普通的加载数据
    交叉验证需要用另外的
    :param samples_num: -1  使用全部样本
    :param division_rate: -1 返回全部   0.7拆分训练集和测试集
    :param file_name: 使用数据集的文件名
    :param di:
    :return: 拆分好的训练集测试集
    """
    data = get_data(file_name)
    X, y = data[:, :-1], data[:, -1]
    train_X, train_y, test_X, test_y = data_split(X, y, division_rate, samples_num)
    if di is not None:
        di.update_item_origin(file_name, train_X.shape[1], samples_num, division_rate)

    return train_X, train_y, test_X, test_y


def load_data_hierarchically(file_name, di=None, division_rate=1, samples_num=-1):
    """
    普通的加载数据
    交叉验证需要用另外的
    :param samples_num: -1  使用全部样本
    :param division_rate: -1 返回全部   0.7拆分训练集和测试集
    :param file_name: 使用数据集的文件名
    :param di:
    :return: 拆分好的训练集测试集
    """
    data = get_data(file_name)
    X, y = data[:, :-1], data[:, -1]
    a, p = np.unique(y, return_inverse=True)
    split = int(samples_num / len(a))  # 分成类别种类份  每份需要多少样本
    idx_list = []
    for idx in a:
        idx_list = idx_list + list(np.argwhere(y == idx))[:split]
    train_X, train_y, test_X, test_y = data_split(X[idx_list, :].reshape(-1, X.shape[1]), y[idx_list], division_rate,
                                                  samples_num)
    if di is not None:
        di.update_item_origin(file_name, train_X.shape[1], samples_num, division_rate)

    return train_X, train_y, test_X, test_y


def data_split(x, y, rate=0.7, samples_num=-1, shuffle=True):
    """
    数据的处理  生成测试集和训练集
    :param x: 特征
    :param y: 类标签
    :param rate: 分割比
    :param samples_num: 选取的样本数
    :param shuffle: 是否随机
    :return:
    """
    index = [i for i in range(len(x))]
    if shuffle:
        t = time.time()
        random.seed(t)
        random.shuffle(index)
    if samples_num != -1:
        index = index[:samples_num]
    x = x[index]
    y = y[index]

    split_index = int(x.shape[0] * rate)
    train_x, train_y = x[:split_index], y[:split_index]
    test_x, test_y = x[split_index:], y[split_index:]

    # get_element(train_x)

    return train_x, train_y, test_x, test_y


def k_flod_data(X, y, k):
    """
    返回k折交叉验证的索引
    :param X:
    :param y:
    :param k:
    :return:
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random.randint(1, 3000))
    a = kf.split(X)
    train_index_list = []
    test_index_list = []
    for i, (train_index, test_index) in enumerate(a):
        train_index_list.append(train_index)
        test_index_list.append(test_index)

    return train_index_list, test_index_list


if __name__ == '__main__':
    pass
