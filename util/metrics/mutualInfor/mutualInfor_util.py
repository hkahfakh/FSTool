#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
由基础度量组合而成的系数等
"""
import numpy as np

from util.metrics.mutualInfor import mutual_info, entropy, joint_entropy, multiple_mi, conditional_mi, \
    normalized_mutual_info


def calc_su(fi, fj):
    mi = mutual_info(fi, fj)
    if mi == 0:
        return 0
    ec = entropy(fj)
    efi = entropy(fi)

    return 2.0 * mi / (ec + efi)  # SU


def calc_su2(fi, fj, fk):
    mi = multiple_mi([fi, fj, fk])
    if mi == 0:
        return 0
    efi = entropy(fi)
    efj = entropy(fj)

    return 2 * mi / (efi + efj)  # SU


def calc_su3(fi, fj, fk):
    mi = multiple_mi([fi, fj, fk])
    if mi == 0:
        return 0
    efi = entropy(fi)
    efj = entropy(fj)
    efk = entropy(fk)

    return 3 * mi / (efi + efj + efk)  # SU


def calc_if(fi, fj, fk):
    """
    三元互信息归一化
    :param fi:
    :param fj:
    :param fk:
    :return:
    """
    hi = entropy(fi)
    hj = entropy(fj)
    hk = entropy(fk)
    Ixyz = multiple_mi([fi, fj, fk])
    Ixyz = 2 * Ixyz / (hk + hi + hj)

    return Ixyz


def calc_icc(fi, fj):
    mi = mutual_info(fi, fj)
    if mi == 0:
        return 0
    j_fic = joint_entropy([fi, fj])
    j_fic = max(j_fic, np.finfo('float64').eps)

    return mi / j_fic


def calc_icc3(fi, fj, fk):
    mi = multiple_mi([fi, fj, fk])
    if mi == 0:
        return 0
    j_fic = joint_entropy([fi, fj, fk])
    j_fic = max(j_fic, np.finfo('float64').eps)

    return mi / j_fic


def calc_mic(fi, fj):
    return normalized_mutual_info(fi, fj)


def calc_mic3(fi, fj, fk):
    mi = multiple_mi([fi, fj, fk])
    if mi == 0:
        return 0
    h1 = entropy(fi)
    h2 = entropy(fj)
    h3 = entropy(fk)
    return mi / min([h1, h2, h3])


def calc_icr(fi, fj):
    """
    互信息与熵的比值
    :param fi:
    :param fj:
    :return:
    """
    mi = mutual_info(fi, fj)
    if mi == 0:
        return 0
    ec = entropy(fj)
    icr = mi / ec
    return icr


def calc_icfr(fi, fj):
    mi = mutual_info(fi, fj)
    if mi == 0:
        return 0
    ec = entropy(fj)
    efi = entropy(fi)
    icr = mi / ec + mi / efi
    return icr


def calc_cratio(fi, fj, c):
    """
    出自drjmi
    :param fi:
    :param fj:
    :return:
    """
    a = conditional_mi([fi, c], fj) - mutual_info(fi, c)
    b = entropy(fi) + entropy(c)
    cratio = 2 * (a / b)
    return cratio


def calc_iw(fi, fj, c):
    a = conditional_mi([fi, c], fj) - mutual_info(fi, c)
    b = entropy(fi) + entropy(fj)
    iw = 1 + (a / b)
    return iw


def calc_iw2(fi, fj, c):
    a = conditional_mi([fi, c], fj) - mutual_info(fi, c)
    b = entropy(fi) + entropy(fj)
    iw = 2 * (a / b)
    return iw


def calc_iw3(fi, fj, c):
    a = conditional_mi([fi, c], fj) - mutual_info(fi, c)
    b = joint_entropy([fi, fj])
    iw = (a / b)
    return iw


def calc_dw(fi, fj, c):
    a = conditional_mi([fi, c], fj) - mutual_info(fi, c)
    b = entropy(fi) + entropy(c)
    dw = 2 * (a / b)
    return dw


def calc_nic(fi, fj, c):
    a = multiple_mi([fi, fj, c])
    one = mutual_info(fi, c)
    two = mutual_info(fj, c)
    three = mutual_info(fi, fj)
    b = one + two + three
    nic = 3 * (a / b)
    return nic


def calc_ijic(fi, fj, c):
    a = multiple_mi([fi, fj, c])
    one = mutual_info(fi, c)
    two = mutual_info(fj, c)
    three = mutual_info(fi, fj)
    b = one + two
    nic = 3 * (a / b)
    return nic


def calc_relation_score(fi, fj, ruler=None):
    ruler = ruler.lower()
    if ruler == 'icfr':
        mi = calc_icfr(fi, fj)
    elif ruler == 'su':
        mi = calc_su(fi, fj)
    elif ruler == 'icc':
        mi = calc_icc(fi, fj)
    elif ruler == 'icr':
        mi = calc_icr(fi, fj)
    elif ruler == 'mi':
        mi = mutual_info(fi, fj)
    elif ruler == 'mic':
        mi = calc_mic(fi, fj)
    elif ruler == 'nmi':
        mi = normalized_mutual_info(fi, fj)
    else:
        raise ValueError("mi_util里没有这个关系得分")
    return mi


def calc_fc_score_list(F, C, ruler):
    """
    计算每个特征和类标签的互信息指标
    :return: 返回每个特征的价值 numpy数组
    """
    if ruler is None:
        raise ValueError("关系得分的关系不能为空")

    fc_mi = []

    for i in range(F.shape[1]):
        fi = F[:, i]
        mi = calc_relation_score(fi, C, ruler)
        fc_mi.append(mi)
    fc_mi = np.array(fc_mi, dtype="float64")
    return fc_mi
