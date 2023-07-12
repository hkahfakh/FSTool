#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
给出支持的算法和数据集
对给出的进行确认
"""
import os

from .decorate import working_dir


def check_algorithm(name_list):
    from featuresAlgorithm import get_algorithm
    for algo_name in name_list:
        assert get_algorithm(algo_name), algo_name + "特征选择算法不存在"


@working_dir()
def check_dataset(name_list):
    base_path = "./dataSet/"
    for file_name in name_list:
        if file_name.find("mat") != -1:
            file_path = (base_path + "mat/" + file_name)
        elif file_name.find("dis") != -1:
            file_path = (base_path + "discrete/" + file_name)
        elif file_name.find("origin") != -1:
            file_path = (base_path + "origin/" + file_name)
        else:
            file_path = (base_path + "normal/" + file_name)
        assert (os.path.exists(file_path)), file_name + '数据集文件不存在'
    return name_list


def inspection():
    """
    检查使用的数据集和算法是否存在
    :return:
    """
    check_dataset(get_formal_dataset_list())
    check_dataset(get_dis_dataset_list())
    check_dataset(get_normal_dataset_list())
    print("Datasets inspection success")

    check_algorithm(get_algorithm_name_list())
    check_algorithm(get_temp_algorithm_name_list())
    print("Algorithm inspection success")


def get_dis_dataset_list():
    """
    所有离散化的数据集
    :return:
    """
    dis_list = ["lymphography_dis.npy", "spambase_dis.npy", "spectf_dis.npy",
                "optdigits_dis.npy", "pendigits_dis.npy", "wdbc_dis.npy", "zoo_dis.npy",
                "ionosphere_dis.npy", "waveform_dis.npy",

                "semeion_dis.npy",
                "movement_dis.npy", "synthetic_dis.npy",
                "arrhythmia_dis.npy", "clean1_dis.npy", "clean2_dis.npy",
                "colon_dis.npy", "lymphoma_dis.npy",
                "basehock_dis.npy", "pcmac_dis.npy", "relathe_dis.npy", "lung-discrete_dis.npy",
                "warpAR10P_dis.npy", "warpPIE10P_dis.npy",
                "lung_dis.npy", "madelon_dis.npy",
                "usps_dis.npy", "mfeat-fac_dis.npy", "TOX171_dis.npy",
                "mfeat-zer_dis.npy", "mfeat-pix_dis.npy", "nci9_dis.npy",
                "prostate-ge_dis.npy", "glioma_dis.npy", "cllsub111_dis.npy",
                "smkcan187_dis.npy", "arcene_dis.npy", "orlraws10P_dis.npy", "gli85_dis.npy"
                ]
    # return get_formal_dataset_list()
    return dis_list


def get_normal_dataset_list():
    """
    未离散化的数据集
    不常用
    :return:
    """
    normal_list = ["glass.npy", "lymphography_finally.npy", "spambase_origin.npy", "spect_finally.npy",
                   "spectf_finally.npy", "optdigits_finally.npy", "pendigits_finally.npy", "wine.data.npy",
                   "wdbc.npy", "zoo_finally.npy", "ionosphere_finally.npy"]
    return normal_list


def get_algorithm_name_list():
    """
    输出数据到xls  所对应的位置
    """
    # algorithm_name_list = ["mim", "mrmr", "mifs", "drjmim", "twofs", "iwfs", "dwfs", "jmim", "ucrfs", "mrmd", "fcbf",
    #                        "fcbfcfs", "ffsg", "mdrmr", "samb"]
    algorithm_name_list = ["mim", "mrmr", "drjmim", "iwfs", "dwfs", "jmim", "ucrfs", "mrmd", "fcbf", "fcbfcfs"]
    return algorithm_name_list


def get_classifier_list():
    """
    返回支持的分类器
    :return:
    """
    # classifier_name = ["svm", "knn", 'gnb', "dt", "rf", "lr", "mlp"]
    classifier_name = ["svm", "knn", "rf", "lr", "mlp"]
    # classifier_name = ['svm', 'knn', 'gnb', 'rf']
    # classifier_name = ['svm', 'knn', 'gnb']
    # classifier_name = ['rr']
    return classifier_name


"""
下面支持修改
"""


def get_one_dataset_name():
    """
    返回一个数据集
    :return:
    """
    # 普通数据集
    # file_name = "glass.npy"
    # file_name = "lymphography_finally.npy"
    # file_name = "spambase_finally.npy"
    # file_name = "spect_finally.npy"
    # file_name = "spectf_finally.npy"
    # file_name = "optdigits_finally.npy"
    # file_name = "pendigits_finally.npy"
    # file_name = "wine.data.npy"
    # file_name = "wdbc.npy"
    # file_name = "zoo_finally.npy"
    # file_name = "ionosphere_finally.npy"

    # file_name = "arrhythmia_finally.npy"
    # file_name = "clean1_finally.npy"
    # file_name = "clean2_finally.npy"
    # file_name = "isolet5_finally.npy"
    # file_name = "synthetic_finally.npy"
    # file_name = "lung-cancer_finally.npy"
    # file_name = "semeion_finally.npy"

    # asu数据集
    # file_name = "warpAR10P.npy"
    # file_name = "warpPIE10P.npy"

    # 表现差
    # file_name = "isolet_dis.npy"
    # file_name = "orl_dis.npy"
    # file_name = "yale_dis.npy"
    # 太好没有区分度
    # file_name = "allaml_dis.npy"
    # file_name = "leukemia_dis.npy"
    # 开始最高，之后下跌
    # file_name = "spect_dis.npy"
    # file_name = "lung-cancer.npy"

    # 离散化数据集
    # file_name = "glass_dis.npy"
    # file_name = "lymphography_dis.npy"
    # file_name = "spambase_dis.npy"
    # file_name = "spectf_dis.npy"
    # file_name = "optdigits_dis.npy"
    # file_name = "pendigits_dis.npy"
    # file_name = "wine_dis.npy"
    # file_name = "wdbc_dis.npy"
    # file_name = "zoo_dis.npy"
    # file_name = "ionosphere_dis.npy"
    # file_name = "waveform_dis.npy"

    # file_name = "lung-cancer_dis.npy"
    # file_name = "semeion_dis.npy"
    # file_name = "movement_dis.npy"
    # file_name = "synthetic_dis.npy"
    # file_name = "synthetic2_dis.npy"
    # file_name = "arrhythmia_dis.npy"
    # file_name = "clean1_dis.npy"
    # file_name = "clean2_dis.npy"

    # file_name = "colon_dis.npy"
    # file_name = "lymphoma_dis.npy"
    # file_name = "basehock_dis.npy"
    # file_name = "pcmac_dis.npy"
    # file_name = "relathe_dis.npy"
    # file_name = "lung-discrete_dis.npy"
    # file_name = "warpAR10P_dis.npy"
    # file_name = "warpPIE10P_dis.npy"
    # file_name = "lung_dis.npy"
    # file_name = "madelon_dis.npy"
    # file_name = "usps_dis.npy"
    # file_name = "mfeat-fac_dis.npy"
    # file_name = "TOX171_dis.npy"
    # file_name = "mfeat-zer_dis.npy"
    # file_name = "mfeat-pix_dis.npy"
    # file_name = "nci9_dis.npy"
    # file_name = "prostate-ge_dis.npy"
    # file_name = "glioma_dis.npy"
    # file_name = "cllsub111_dis.npy"
    file_name = "smkcan187_dis.npy"
    # file_name = "arcene_dis.npy"
    # file_name = "orlraws10P_dis.npy"
    # file_name = "gli85.npy"

    # file_name = "ORL.mat"
    # file_name = "clean2_dis.npy"
    # file_name = "leukemia.npy"
    # file_name = "isolet_dis.npy"
    # file_name = "prostate-ge.npy"
    return file_name


def get_formal_dataset_list():
    """
    最终数据集
    这些数据集现象明显
    本次任务使用的数据
    :return:
    """
    formal_list = [
        "spambase_dis.npy",
        "optdigits_dis.npy",
        "pendigits_dis.npy",
        "wdbc_dis.npy",
        "ionosphere_dis.npy",
        "movement_dis.npy",
        "synthetic_dis.npy",
        "arrhythmia_dis.npy",
        "clean1_dis.npy",
        "lymphoma_dis.npy",
        "basehock_dis.npy",
        "pcmac_dis.npy",
        "relathe_dis.npy",
        "lung-discrete_dis.npy",
        "warpAR10P_dis.npy",
        "warpPIE10P_dis.npy",
        "lung_dis.npy",
        "usps_dis.npy",
        "mfeat-fac_dis.npy",
        "mfeat-zer_dis.npy",
        "mfeat-pix_dis.npy",
        "prostate-ge_dis.npy",
        "glioma_dis.npy",
        "cllsub111_dis.npy",
        "smkcan187_dis.npy",
        "arcene_dis.npy",
    ]
    return formal_list


def get_temp_algorithm_name_list():
    """
    当前任务要使用哪些算法
    :return:
    """
    algorithm_name_list = [
        "mrmd"]
    return algorithm_name_list
