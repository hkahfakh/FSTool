#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
互信息基础度量实现
"""
import numpy as np
from math import log2
from scipy import sparse as sp
from sklearn.metrics import mutual_info_score
from itertools import combinations


def _generalized_average(U, V, average_method):
    """Return a particular mean of two numbers."""
    if average_method == "min":
        return min(U, V)
    elif average_method == "geometric":
        return np.sqrt(U * V)
    elif average_method == "arithmetic":
        return np.mean([U, V])
    elif average_method == "max":
        return max(U, V)
    else:
        raise ValueError("'average_method' must be 'min', 'geometric', "
                         "'arithmetic', or 'max'")


def _contingency_matrix1(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.

        .. versionadded:: 0.18

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


def _contingency_matrix_temp(condition, test):
    """
    多个条件的列联表
    condition
    :param condition:是个列表，里面每个元素是一列特征
    :param test:
    :return:
    """
    n_bin = []
    idx = []

    tests, test_idx = np.unique(test, return_inverse=True)
    n_tests = tests.shape[0]
    idx.append(test_idx)
    n_bin.append(n_tests)

    for c in condition[::-1]:
        classes, class_idx = np.unique(c, return_inverse=True)
        n_classes = classes.shape[0]
        idx.append(class_idx)
        n_bin.append(n_classes)

    t = np.histogramdd(idx, bins=n_bin, )
    return t[0]


def _contingency_matrix(condition, test):
    """
    多个条件的列联表

    :param condition: 是个矩阵  每一列是个特征
    :param test:
    :return:
    """
    if len(condition.shape) == 1:
        condition = condition.reshape(-1, 1)  # 单个特征的处理
    n_bin = []
    idx = []

    tests, test_idx = np.unique(test, return_inverse=True)
    n_tests = tests.shape[0]
    idx.append(test_idx)
    n_bin.append(n_tests)

    for c in condition.T[::-1]:
        classes, class_idx = np.unique(c, return_inverse=True)
        n_classes = classes.shape[0]
        idx.append(class_idx)
        n_bin.append(n_classes)

    t = np.histogramdd(idx, bins=n_bin, )
    return t[0]


def _contingency_matrix2(condition):
    """
    多个条件和类标签的列联表
    :param condition: 列表  每个元素是一列特征
    :return:
    """
    n_bin = []
    idx = []

    for c in condition[::-1]:
        classes, class_idx = np.unique(c, return_inverse=True)
        n_classes = classes.shape[0]
        idx.append(class_idx)
        n_bin.append(n_classes)

    t = np.histogramdd(idx, bins=n_bin, )
    # t = cupy.histogramdd(cupy.array(idx).T, bins=cupy.array(n_bin), )
    return t[0]


def entropy(labels):
    """
    正确
    Calculates the entropy for a labeling.
    Parameters
    ----------
    labels : int array, shape = [n_samples]
        The labels
    """
    if len(labels) == 0:
        return 1.0

    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)  # 总数
    # log2(a / b) should be calculated as log2(a) - log2(b) for possible loss of precision
    # 直接数组每一位都除pi_sum，变成了概率
    return -np.sum((pi / pi_sum) * (np.log2(pi) - np.log2(pi_sum)))
    # return ne.evaluate("sum((pi / pi_sum) * (log(pi) - log2(pi_sum)),0)")


def conditional_entropy(z, condition):
    """
    多个条件的条件熵

    二次确认正确
    对应 p(x,y,z)*[log2(p(xy))-log2(P(x,y,z)])

    :param condition:里面都是一个个特征  注意里面应当是一个特征作为一个元素
    :param z:
    :return:
    """
    contingency = _contingency_matrix2(condition + [z])

    nz_pos = np.nonzero(contingency)  # nz是非零坐标
    nz_val = contingency[nz_pos]  # 取出所有非零数

    contingency_sum = contingency.sum()
    contingency_nm = nz_val / contingency_sum  # xyz确定时的概率
    log_contingency_nm = np.log2(nz_val)  # 对列联表中所有非零数求对数

    p_condition = contingency.sum(axis=0)  # 这几个应当是投影到对应轴的频数不是概率
    outer = p_condition[nz_pos[1:]]  # 提取p1中nz1位置的值  返回
    log_outer = np.log2(outer) - log2(p_condition.sum())

    cond_ent = (contingency_nm *  # p(x,y,z)
                log_outer -  # log2(p(xy))
                contingency_nm *  # p(x,y,z)
                (log_contingency_nm - log2(contingency_sum))  # logP(x,y,z)
                )
    return cond_ent.sum()


def joint_entropy(arg):
    """
    任意多个特征联合熵
    二次确认正确
    https://zhuanlan.zhihu.com/p/35379531
    :param arg:
    :param temp:
    :param z:
    :return:
    """
    contingency = _contingency_matrix2(arg)

    nz = np.nonzero(contingency)  # nz是非零坐标
    nz_val = contingency[nz]  # 取出所有非零数

    contingency_sum = contingency.sum()

    contingency_nm = nz_val / contingency_sum  # xyz确定时的概率
    log_contingency_nm = np.log2(nz_val)  # 对列联表中所有非零数求对数

    joint_ent = (contingency_nm *  # p(x,y,z)
                 (log_contingency_nm - log2(contingency_sum))  # logP(x,y,z)
                 )
    return -joint_ent.sum()


def conditional_joint_entropy(x, y, condition):
    """
    条件联合熵
    :param x:
    :param y:
    :param condition:
    :return:
    """
    return joint_entropy([x, y, condition]) - entropy(condition)


def mutual_info(labels_true, labels_pred):
    """
    来源于metrics库
    绝对正确
    :param labels_true:
    :param labels_pred:
    :return:
    """
    contingency = _contingency_matrix1(labels_true, labels_pred, sparse=False)

    # For an array
    nzx, nzy = np.nonzero(contingency)
    nz_val = contingency[nzx, nzy]

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log2(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    t1 = pi.take(nzx).astype(np.int64, copy=False)
    t2 = pj.take(nzy).astype(np.int64, copy=False)
    outer = (t1 * t2)
    log_outer = -np.log2(outer) + log2(pi.sum()) + log2(pj.sum())  # 求的ij同时满足的概率log(px*py)
    mi = (contingency_nm *  # p(x,y)
          (log_contingency_nm - log2(contingency_sum)) +  # log2(p(x,y))
          contingency_nm *  # p(x,y)
          log_outer)  # -log(px*py)
    s = mi.sum()
    if s < 0:
        s = 0
    return s


def normalized_mutual_info(labels_true, labels_pred):
    """
    归一化互信息
    :param labels_true:
    :param labels_pred:
    :return:
    """
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1 or
            classes.shape[0] == clusters.shape[0] == 0):
        return 1.0
    contingency = _contingency_matrix1(labels_true, labels_pred, sparse=False)
    # Calculate the MI for the two clusterings
    mi = mutual_info(labels_true, labels_pred, )
    # Calculate the expected value for the mutual information
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    normalizer = _generalized_average(h_true, h_pred, "min")
    # Avoid 0.0 / 0.0 when either entropy is zero.
    normalizer = max(normalizer, np.finfo('float64').eps)
    nmi = mi / normalizer
    return nmi


def mutual_info_coefficient(labels_true, labels_pred):
    """
    互信息系数MIC
    :param labels_true:
    :param labels_pred:
    :return:
    """
    from minepy import MINE
    mine = MINE()
    mine.compute_score(labels_true, labels_pred)
    m = mine.mic()

    contingency = _contingency_matrix1(labels_true, labels_pred, sparse=False)

    # For an array
    nzx, nzy = np.nonzero(contingency)
    nz_val = contingency[nzx, nzy]

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log2(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    t1 = pi.take(nzx).astype(np.int64, copy=False)
    t2 = pj.take(nzy).astype(np.int64, copy=False)
    outer = (t1 * t2)
    log_outer = -np.log2(outer) + log2(pi.sum()) + log2(pj.sum())  # 求的ij同时满足的概率log(px*py)
    mi = (contingency_nm *  # p(x,y)
          (log_contingency_nm - log2(contingency_sum)) +  # log2(p(x,y))
          contingency_nm *  # p(x,y)
          log_outer)  # -log(px*py)
    t = mutual_info_score(labels_true, labels_pred)
    mi = mi.sum()
    a = min(contingency.shape)
    b = log2(a)
    mic = mi / b
    return m


def conditional_mi(m, condition):
    """
    条件互信息

    :param m: 两个特征
    :param condition: 多个特征 多个条件
    :return:
    """
    if type([1]) != type(condition):  # 多个condition  才是list
        contingency = _contingency_matrix2(m + [condition])
    else:
        contingency = _contingency_matrix2(m + condition)

    nz_pos = np.nonzero(contingency)  # nz是非零坐标
    nz_val = contingency[nz_pos]  # 取出所有非零数

    contingency_sum = contingency.sum()
    contingency_nm = nz_val / contingency_sum  # xyz确定时的概率
    log_contingency_nm = np.log2(nz_val)  # 对列联表中所有非零数求对数

    # Don't need to calculate the full outer product, just for non-zeroes

    temp = nz_pos[0:-2] + nz_pos[-1:]
    p_x_condition = contingency.sum(axis=-2)
    outer = p_x_condition[temp]
    log_pxz = np.log2(outer) - log2(p_x_condition.sum())

    temp = nz_pos[:-1]
    p_y_condition = contingency.sum(axis=-1)
    outer = p_y_condition[temp]
    log_pyz = np.log2(outer) - log2(p_y_condition.sum())

    a = tuple(range(1, len(contingency.shape)))
    p_condition = np.ravel(contingency.sum(axis=a))
    outer = p_condition.take(nz_pos[0]).astype(np.int64, copy=False)
    log_pz = np.log2(outer) - log2(p_condition.sum())

    # p(x,y,z)*[log(p(x,y))-log(p(x,y,z))] 用的好像不是这个公式
    cond_mi = (contingency_nm *  # p(x,y,z)
               (log_pz + (log_contingency_nm - log2(contingency_sum))) -
               (contingency_nm *  # p(x,y,z)
                (log_pxz + log_pyz))
               )
    return cond_mi.sum()


def conditional_mi_coefficient(m, condition):
    """
    条件互信息系数

    :param m: 两个特征
    :param condition: 多个特征  多个条件
    :return:
    """
    if type([1]) != type(condition):  # 多个condition  才是list
        contingency = _contingency_matrix2(m + [condition])
    else:
        contingency = _contingency_matrix2(m + condition)

    nz_pos = np.nonzero(contingency)  # nz是非零坐标
    nz_val = contingency[nz_pos]  # 取出所有非零数

    contingency_sum = contingency.sum()
    contingency_nm = nz_val / contingency_sum  # xyz确定时的概率
    log_contingency_nm = np.log2(nz_val)  # 对列联表中所有非零数求对数

    # Don't need to calculate the full outer product, just for non-zeroes

    temp = nz_pos[0:-2] + nz_pos[-1:]
    p_x_condition = contingency.sum(axis=-2)
    outer = p_x_condition[temp]
    log_pxz = np.log2(outer) - log2(p_x_condition.sum())

    temp = nz_pos[:-1]
    p_y_condition = contingency.sum(axis=-1)
    outer = p_y_condition[temp]
    log_pyz = np.log2(outer) - log2(p_y_condition.sum())

    a = tuple(range(1, len(contingency.shape)))
    p_condition = np.ravel(contingency.sum(axis=a))
    outer = p_condition.take(nz_pos[0]).astype(np.int64, copy=False)
    log_pz = np.log2(outer) - log2(p_condition.sum())

    cond_mi = (contingency_nm *  # p(x,y,z)
               (log_pz + (log_contingency_nm - log2(contingency_sum))) -
               (contingency_nm *  # p(x,y,z)
                (log_pxz + log_pyz))
               )
    cond_mi = cond_mi.sum()
    cond_mic = cond_mi / log2(min(contingency.shape))
    return cond_mic


def joint_mi(arg, C):
    """
    联合互信息
    arg组成的联合熵 与C的互信息
    :param arg:
    :return:
    """
    a = mutual_info(arg[0], C)
    b = conditional_mi([arg[1], C], [arg[0]])
    return a + b

    contingency = _contingency_matrix2(arg)

    nz_pos = np.nonzero(contingency)  # nz是非零坐标
    nz_val = contingency[nz_pos]  # 取出所有非零数

    contingency_sum = contingency.sum()
    log_contingency_nm = np.log2(nz_val)  # 对列联表中所有非零数求对数
    contingency_nm = nz_val / contingency_sum  # xyz确定时的概率
    # Don't need to calculate the full outer product, just for non-zeroes

    log_sum = 0
    outer = np.ones(nz_pos[0].shape[0])
    ax = list(range(len(nz_pos)))
    for i in range(len(contingency.shape)):
        temp = tuple(ax[:i] + ax[i + 1:])
        p_i = np.ravel(contingency.sum(axis=temp))
        log_sum = log_sum + log2(p_i.sum())
        outer = outer * p_i.take(nz_pos[i]).astype(np.int64, copy=False)
    log_outer = -np.log2(outer) + log_sum

    mi = (contingency_nm *  # p(x,y,z)对
          (log_contingency_nm - log2(contingency_sum)) +  # logP(x,y,z)对
          contingency_nm *  # p(x,y,z)对
          log_outer  # -log(p(x)p(y)p(z))
          )
    return mi.sum()


def multiple_mi(arg):
    """
    三元互信息
    I(X；Y；Z)
    :param arg:
    :return:
    """
    contingency = _contingency_matrix2(arg)

    nz_pos = np.nonzero(contingency)  # nz是非零坐标
    nz_val = contingency[nz_pos]  # 取出所有非零数

    contingency_sum = contingency.sum()
    contingency_nm = nz_val / contingency_sum  # xyz确定时的概率
    log_contingency_nm = np.log2(nz_val)  # 对列联表中所有非零数求对数

    temp = nz_pos[:-1]
    p_condition = contingency.sum(axis=2)  # 这几个应当是投影到对应轴的频数不是概率
    outer = p_condition[temp]  # 提取p1中nz1位置的值  返回
    log_pyz = np.log2(outer) - log2(p_condition.sum())

    temp = nz_pos[0:-2] + nz_pos[-1:]
    p_condition = contingency.sum(axis=1)  # 这几个应当是投影到对应轴的频数不是概率
    outer = p_condition[temp]  # 提取p1中nz1位置的值  返回
    log_pxz = np.log2(outer) - log2(p_condition.sum())

    temp = nz_pos[1:]
    p_condition = contingency.sum(axis=0)  # 这几个应当是投影到对应轴的频数不是概率
    outer = p_condition[temp]  # 提取p1中nz1位置的值  返回
    log_pxy = np.log2(outer) - log2(p_condition.sum())

    a = tuple([0, 1])
    p_condition = np.ravel(contingency.sum(axis=a))
    outer = p_condition.take(nz_pos[2]).astype(np.int64, copy=False)
    log_px = np.log2(outer) - log2(p_condition.sum())

    a = tuple([0, 2])
    p_condition = np.ravel(contingency.sum(axis=a))
    outer = p_condition.take(nz_pos[1]).astype(np.int64, copy=False)
    log_py = np.log2(outer) - log2(p_condition.sum())

    a = tuple(range(1, len(contingency.shape)))
    p_condition = np.ravel(contingency.sum(axis=a))
    outer = p_condition.take(nz_pos[0]).astype(np.int64, copy=False)
    log_pz = np.log2(outer) - log2(p_condition.sum())

    muti_mi = (contingency_nm *  # p(x,y,z)
               (log_pxy + log_pxz + log_pyz) -
               (contingency_nm *  # p(x,y,z)
                (log_px + log_py + log_pz + (log_contingency_nm - log2(contingency_sum))))
               )
    muti_mi = muti_mi.sum()
    return -muti_mi


def formula_check():
    """
    计算各概念之间的关系是否正确
    :return:
    """
    from util.data_process import get_data
    d = get_data("arrhythmia_finally.npy")
    d = d.astype(np.float64)

    # h(x, y, z) = h(x, y) + h(z | x, y)
    print("h(x, y, z) = h(x, y) + h(z | x, y)")
    print(joint_entropy([d[:, 2], d[:, 3], d[:, -1]]))
    hxy = joint_entropy([d[:, 2], d[:, 3]])
    hzxy = conditional_entropy(d[:, -1], [d[:, 2], d[:, 3]])
    print("{}={}+{}".format(hxy + hzxy, hxy, hzxy))

    # i(xy) = h(x) - h(x | y)
    print("i(xy) = h(x) - h(x | y)")
    print(mutual_info(d[:, 2], d[:, 3]))
    hx = entropy(d[:, 2])
    hxy = conditional_entropy(d[:, 2], [d[:, 3]])
    print("{}={}-{}".format(hx - hxy, hx, hxy))

    # h(x, y) = h(x) + h(y | x)
    print("h(x, y) = h(x) + h(y | x)")
    print(joint_entropy([d[:, 2], d[:, -1]]))
    hx = entropy(d[:, 2])
    hyx = conditional_entropy(d[:, -1], [d[:, 2]])
    print("{}={}+{}".format(hx + hyx, hx, hyx))

    # I(x;y|z) = h(xz) + h(yz) - h(xyz) - h(z)
    print("I(x;y|z) = h(xz) + h(yz) - h(xyz) - h(z)")
    print(conditional_mi([d[:, 1], d[:, 2]], d[:, -1]))
    hxz = joint_entropy([d[:, 1], d[:, -1]])
    hyz = joint_entropy([d[:, 2], d[:, -1]])
    hxyz = joint_entropy([d[:, 1], d[:, 2], d[:, -1]])
    hz = entropy(d[:, -1])
    print("{}={}+{}-{}-{}".format(hxz + hyz - hxyz - hz, hxz, hyz, hxyz, hz))

    # h(xyz) = h(x) + h(y) + h(z) - i(xy) - i(yz) - i(xz) + i(xyz)
    print("h(xyz) = h(x) + h(y) + h(z) - i(xy) - i(yz) - i(xz) - i(xyz)")
    print(joint_entropy([d[:, 1], d[:, 2], d[:, -1]]))
    hx = entropy(d[:, 1])
    hy = entropy(d[:, 2])
    hz = entropy(d[:, -1])
    ixy = mutual_info(d[:, 1], d[:, 2])
    iyz = mutual_info(d[:, 2], d[:, -1])
    ixz = mutual_info(d[:, 1], d[:, -1])
    ixyz = multiple_mi([d[:, 1], d[:, 2], d[:, -1]])
    print("{}={}+{}+{}-{}-{}-{}-{}".format(hx + hy + hz - ixy - iyz - ixz - ixyz, hx, hy, hz, ixy, iyz, ixz, ixyz))

    # I(x;y;z) = I(x;y) - I(x;y|z)
    print("I(x;y;z) = I(x;y|z) - I(x;y)")
    print(multiple_mi([d[:, 1], d[:, 2], d[:, -1]]))
    ixyz = conditional_mi([d[:, 1], d[:, 2]], d[:, -1])
    ixy = mutual_info(d[:, 1], d[:, 2])
    print("{}={}-{}".format(ixyz - ixy, ixy, ixyz))

    # I(x;y;z) = I(y;z|x) - I(x;z)
    print("I(x;y;z) = I(y;z|x) - I(x;z)")
    print(multiple_mi([d[:, 1], d[:, 2], d[:, -1]]))
    iyzx = conditional_mi([d[:, 2], d[:, -1]], d[:, 1])
    iyz = mutual_info(d[:, 2], d[:, -1])
    print("{}={}-{}".format(iyzx - iyz, iyzx, iyz))

    # I(x,y;z) = I(x;z) + I(y;z|x)
    print("I(x,y;z) = I(x;z) + I(y;z|x)")
    print(joint_mi([d[:, 1], d[:, 2]], d[:, -1]))
    ixy = mutual_info(d[:, 1], d[:, -1])
    ixyz = conditional_mi([d[:, 2], d[:, -1]], d[:, 1])
    print("{}={}+{}".format(ixy + ixyz, ixy, ixyz))


if __name__ == '__main__':
    # formula_check()
    from util.data_process import get_data

    d = get_data("arrhythmia_finally.npy")
    d = d.astype(np.float64)

    c = mutual_info([d[:, 1]], d[:, 2])
    e = mutual_info([d[:, 2]], d[:, -1])

    # contingency1 = _contingency_matrix_temp([d[:, 1], d[:, 2]], d[:, -1])
    # contingency2 = _contingency_matrix(d[:, 1:3], d[:, -1])
    # contingency3 = _contingency_matrix(d[:, 1], d[:, -1])
