#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.io as sio


def mat2other(file_path, file_type):
    save_name = file_path[:-4]
    mat = sio.loadmat(file_path)
    X = mat['X']  # data
    X = X.astype(float)
    y = mat['Y']  # label
    y = np.array([y[:, 0]]).T
    data = np.hstack((X, y))

    if file_type == "csv":  # 保存为csv
        dfdata = pd.DataFrame(data=data)
        dfdata.to_csv(save_name + '.csv', index=False)

    elif file_type == "npy":  # 保存为npy
        np.save(save_name, data)


def csv2other(file_path, file_type):
    save_name = file_path
    # save_name = file_path[:-4]
    data = pd.read_csv(file_path, header=None, low_memory=False)  # 编码默认UTF-8，若乱码自行更改
    data = data.values

    a = np.array(data)
    if file_type == "npy":  # 保存为npy
        np.save(save_name, data)
    return a


if __name__ == '__main__':
    mat2other("../../../dataSet/mat/GLI_85.mat", "npy")
    # csv2other("../../../dataSet/pure/wdbc.data", "npy")
