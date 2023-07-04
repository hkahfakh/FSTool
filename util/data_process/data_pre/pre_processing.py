#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np
from .discrete import Discretization_EqualFrequency, standardization
from ..data_gen.getData import get_element, get_data


def kddcup_preprocessing(data, save_flag=1):
    protocol = ['tcp', 'icmp', 'udp']
    service = ['IRC', 'X11', 'Z39_50', 'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf',
               'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs',
               'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http',
               'http_2784', 'http_443', 'http_8001', 'imap4', 'iso_tsap', 'klogin', 'kshell',
               'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns',
               'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2',
               'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp',
               'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i',
               'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois']
    dst_host_srv_rerror_rate = ['normal.', 'back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.',
                                'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.',
                                'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.',
                                'teardrop.', 'warezclient.', 'warezmaster.']
    flag = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    continuous = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 0]

    for i in range(data.shape[0]):
        data[:, 1][i] = protocol.index(data[:, 1][i])
        data[:, 2][i] = service.index(data[:, 2][i])
        data[:, 3][i] = flag.index(data[:, 3][i])
        if dst_host_srv_rerror_rate.index(data[:, -1][i]) == 0:
            data[:, -1][i] = 0
        else:
            data[:, -1][i] = 1

    data = data.astype(np.float64)
    for j in range(len(continuous)):
        if continuous[j] == 1:
            data[:, j] = standardization(data[:, j])
    if save_flag == 1:
        np.save("./dataSet/KDDCup_finally.npy", data)
    get_element(data)
    return data


def hypothyroid_preprocessing(data, save_flag=1):
    # 未完成
    classes = ['hypothyroid' 'negative']
    on_thyroxine = ['?' 'F' 'M']
    ft = ['f' 't']
    ny = ['n' 'y']
    get_element(data)
    continuous = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    for i in range(data.shape[0]):
        data[:, 2][i] = classes.index(data[:, 2][i])
        data[:, 3][i] = on_thyroxine.index(data[:, 3][i])
        data[:, 4][i] = ft.index(data[:, 4][i])

    for j in range(len(continuous)):
        if continuous[j] == 1:
            data[:, j] = standardization(data[:, j])
    if save_flag == 1:
        np.save("./dataSet/hypothyroid_finally.npy", data)
    data = np.delete(data, -1, axis=25)
    get_element(data)
    return data


def arrhythmia_preprocessing(data, save_flag=1):
    # 未完成
    data = np.delete(data, 13, axis=1)

    data[data == '?'] = 0
    data = data.astype("float64")
    if save_flag == 1:
        np.save("./dataSet/arrhythmia_finally.npy", data)

    return data


def mfeat_preprocessing(data, save_flag=1):
    reconstuct_list = []
    for i in data:
        pat1 = '\d+\.\d+'
        info = re.findall(pat1, i[0])
        reconstuct_list.append(info)

    reconstuct_list = np.array(reconstuct_list)
    reconstuct_list = reconstuct_list.astype(dtype='float64')
    b = np.array(
        [0] * 200 + [1] * 200 + [2] * 200 + [3] * 200 + [4] * 200 + [5] * 200 + [6] * 200 + [7] * 200 + [8] * 200 + [
            9] * 200)
    reconstuct_list = np.column_stack((reconstuct_list, b.T))
    if save_flag == 1:
        np.save("./dataSet/mfeat_zer_finally.npy", reconstuct_list)
    return reconstuct_list


def glass_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    a = a.astype("float64")
    a = a[:, 1:]
    if save_flag == 1:
        np.save("./dataSet/glass.npy", a)  # 保存为.npy格式


def optdigits_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    a = a.astype("float64")
    if save_flag == 1:
        np.save("./dataSet/optdigits_finally.npy", a)  # 保存为.npy格式


def onehr_preprocessing(data, save_flag=1):
    """
    onehr数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    a[a == '?'] = 0
    a = a[:, 1:]
    for i in range(a.shape[0]):
        a[:, -1][i] = a[:, -1][i][:-1]
    a = a.astype("float64")
    if save_flag == 1:
        np.save("./dataSet/onehr_finally.npy", a)  # 保存为.npy格式


def spambase_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    a = a.astype("float64")
    a[:, :-1] = Discretization_EqualFrequency(a[:, :-1])
    if save_flag == 1:
        np.save("./dataSet/spambase_finally.npy", a)  # 保存为.npy格式


def wdbc_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    diagnosis = ['M', 'B']
    for i in range(data.shape[0]):
        data[:, 1][i] = diagnosis.index(data[:, 1][i])
    data[:, [-1, 1]] = data[:, [1, -1]]
    data = data[:, 1:]

    get_element(data)
    a = np.array(data)
    a = a.astype("float64")
    if save_flag == 1:
        np.save("./dataSet/wdbc.npy", a)  # 保存为.npy格式


def nslkdd_preprocessing(data, save_flag=1):
    protocol = ['icmp', 'tcp', 'udp']
    service = ['IRC', 'X11', 'Z39_50', 'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf',
               'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs',
               'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http',
               'http_2784', 'http_443', 'http_8001', 'imap4', 'iso_tsap', 'klogin', 'kshell',
               'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns',
               'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2',
               'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp',
               'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i',
               'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois']
    flag = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    dst_host_srv_rerror_rate = ['apache2', 'back', 'buffer_overflow', 'ftp_write', 'guess_passwd',
                                'httptunnel', 'imap', 'ipsweep', 'land', 'loadmodule', 'mailbomb', 'mscan',
                                'multihop', 'named', 'neptune', 'nmap', 'normal', 'perl', 'phf', 'pod',
                                'portsweep', 'processtable', 'ps', 'rootkit', 'saint', 'satan', 'sendmail',
                                'smurf', 'snmpgetattack', 'snmpguess', 'spy', 'sqlattack', 'teardrop',
                                'udpstorm', 'warezclient', 'warezmaster', 'worm', 'xlock', 'xsnoop', 'xterm']

    continuous = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 0]

    for i in range(data.shape[0]):
        data[:, 1][i] = protocol.index(data[:, 1][i])
        data[:, 2][i] = service.index(data[:, 2][i])
        data[:, 3][i] = flag.index(data[:, 3][i])
        if dst_host_srv_rerror_rate.index(data[:, -2][i]) == 0:
            data[:, -2][i] = 0
        else:
            data[:, -2][i] = 1

    data = np.delete(data, -1, axis=1)  # 最后一列不知道啥东西

    data = data.astype(np.float64)

    for j in range(len(continuous)):
        if continuous[j] == 1:
            data[:, j] = standardization(data[:, j])

    if save_flag == 1:
        np.save("./dataSet/NSLKDD_finally.npy", data)
    get_element(data)
    return data


def unsw_preprocessing(data, save_flag=1):
    """
    大多数列  unique后维数太多  需要离散化
    :param data:
    :param save_flag:
    :return:
    """
    proto = ['3pc', 'a/n', 'aes-sp3-d', 'any', 'argus', 'aris', 'arp', 'ax.25', 'bbn-rcc',
             'bna', 'br-sat-mon', 'cbt', 'cftp', 'chaos', 'compaq-peer', 'cphb', 'cpnx',
             'crtp', 'crudp', 'dcn', 'ddp', 'ddx', 'dgp', 'egp', 'eigrp', 'emcon', 'encap',
             'esp', 'etherip', 'fc', 'fire', 'ggp', 'gmtp', 'gre', 'hmp', 'i-nlsp', 'iatp', 'ib',
             'icmp', 'idpr', 'idpr-cmtp', 'idrp', 'ifmp', 'igmp', 'igp', 'il', 'ip', 'ipcomp',
             'ipcv', 'ipip', 'iplt', 'ipnip', 'ippc', 'ipv6', 'ipv6-frag', 'ipv6-no',
             'ipv6-opts', 'ipv6-route', 'ipx-n-ip', 'irtp', 'isis', 'iso-ip', 'iso-tp4',
             'kryptolan', 'l2tp', 'larp', 'leaf-1', 'leaf-2', 'merit-inp', 'mfe-nsp', 'mhrp',
             'micp', 'mobile', 'mtp', 'mux', 'narp', 'netblt', 'nsfnet-igp', 'nvp', 'ospf',
             'pgm', 'pim', 'pipe', 'pnni', 'pri-enc', 'prm', 'ptp', 'pup', 'pvp', 'qnx', 'rdp',
             'rsvp', 'rtp', 'rvd', 'sat-expak', 'sat-mon', 'sccopmce', 'scps', 'sctp', 'dirp',
             'secure-vmtp', 'sep', 'skip', 'sm', 'smp', 'snp', 'sprite-rpc', 'sps', 'srp',
             'st2', 'stp', 'sun-nd', 'swipe', 'tcf', 'tcp', 'tlsp', 'tp++', 'trunk-1',
             'trunk-2', 'ttp', 'udp', 'udt', 'unas', 'uti', 'vines', 'visa', 'vmtp', 'vrrp',
             'wb-expak', 'wb-mon', 'wsn', 'xnet', 'xns-idp', 'xtp', 'zero']
    state = ['ACC', 'CLO', 'CON', 'ECO', 'ECR', 'FIN', 'INT', 'MAS', 'PAR', 'REQ', 'RST', 'TST',
             'TXD', 'URH', 'URN', 'no']
    service = ['-', 'dhcp', 'dns', 'ftp', 'ftp-data', 'http', 'irc', 'pop3', 'radius', 'smtp', 'snmp', 'ssh', 'ssl']
    continuous = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                  1, 0, 1, 1, 1, 1, 1, 1, 1, 0]

    for i in range(data.shape[0]):
        data[:, 4][i] = proto.index(data[:, 4][i])
        data[:, 5][i] = state.index(data[:, 5][i])
        data[:, 13][i] = service.index(data[:, 13][i])

    data = np.delete(data, [26, 27, 28, 29, 30, 31], axis=1)  # 删除抖动和时间戳
    data = data[:, 4:]  # 删除传输ip port
    data = np.delete(data, -2, axis=1)  # 删除攻击类型  因为这个数据集有是否攻击的标签

    data[data == " "] = '0'
    data[data == ""] = '0'

    data = data.astype(np.float64)

    for j in range(len(continuous)):
        if continuous[j] == 1:
            data[:, j] = standardization(data[:, j])
    if save_flag == 1:
        np.save("./dataSet/UNSW_finally.npy", data)
    get_element(data)
    return data


def pendigits_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    a = a.astype("int")
    if save_flag == 1:
        np.save("./dataSet/pendigits_finally.npy", a)  # 保存为.npy格式


def movement_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    a = a.astype("float64")
    if save_flag == 1:
        np.save("./dataSet/movement_finally.npy", a)  # 保存为.npy格式


def spectf_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    a = a.astype("float64")

    a[:, [0, -1]] = a[:, [-1, 0]]
    if save_flag == 1:
        np.save("./dataSet/spectf_finally.npy", a)  # 保存为.npy格式


def spect_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    a = a.astype("float64")

    a[:, [0, -1]] = a[:, [-1, 0]]
    if save_flag == 1:
        np.save("./dataSet/spect_finally.npy", a)  # 保存为.npy格式


def isolet_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    for i in range(a.shape[0]):
        a[:, -1][i] = a[:, -1][i][:-1]

    for j in range(a.shape[1] - 1):
        data[:, j] = standardization(data[:, j])

    a = a.astype("float64")
    # a[:, -1] = a[:, -1].astype("int")
    if save_flag == 1:
        np.save("./dataSet/isolet5_finally.npy", a)  # 保存为.npy格式


def zoo_preprocessing(data, save_flag=1):
    """
    zoo数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    # for i in range(a.shape[0]):
    #     a[:, -1][i] = a[:, -1][i][:-1]
    a = np.delete(a, 0, axis=1)
    a = a.astype("float64")
    if save_flag == 1:
        np.save("./dataSet/zoo_finally.npy", a)  # 保存为.npy格式


def ionosphere_preprocessing(data, save_flag=1):
    """
    zoo数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)

    lable = ['b', 'g']
    for i in range(a.shape[0]):
        a[:, -1][i] = lable.index(a[:, -1][i])

    a = a.astype("float64")
    if save_flag == 1:
        np.save("./dataSet/ionosphere_finally.npy", a)  # 保存为.npy格式


def lymphography_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    a = a.astype("float64")
    a[:, [0, -1]] = a[:, [-1, 0]]
    if save_flag == 1:
        np.save("./dataSet/lymphography_finally.npy", a)  # 保存为.npy格式


def synthetic_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    a = a.astype("float64")
    # b = np.array([0] * 200 +
    #              [1] * 200 +
    #              [2] * 200 +
    #              [3] * 200 +
    #              [4] * 200 +
    #              [5] * 200 +
    #              [6] * 200 +
    #              [7] * 200 +
    #              [8] * 200 +
    #              [9] * 200
    #              )
    b = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100 + [5] * 100)
    a = np.column_stack((a, b.T))
    # a[:, -1] = a[:, -1].astype("int")
    if save_flag == 1:
        np.save("./dataSet/synthetic_finally.npy", a)  # 保存为.npy格式


def splice_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    a = a.astype("float64")

    # a[:, -1] = a[:, -1].astype("int")
    if save_flag == 1:
        np.save("./dataSet/splice_finally.npy", a)  # 保存为.npy格式


def clean1_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    a = a[:, 2:]
    for i in range(a.shape[0]):
        a[:, -1][i] = a[:, -1][i][:-1]
    a = a.astype("int")

    # a[:, -1] = a[:, -1].astype("int")
    if save_flag == 1:
        np.save("./dataSet/clean1_finally.npy", a)  # 保存为.npy格式


def lung_cancer_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)
    # for i in range(a.shape[0]):
    #     a[:, -1][i] = a[:, -1][i][:-1]
    a = a.astype("int")

    # a[:, -1] = a[:, -1].astype("int")
    if save_flag == 1:
        np.save("./dataSet/lung-cancer_finally.npy", a)  # 保存为.npy格式


def semeion_preprocessing(data, save_flag=1):
    """
    glass数据集预处理以及生成npy
    :return:
    """

    a = np.array(data)[:, :266]
    label = []
    for temp in a[:, -10:]:
        label.append(np.where(temp == '1')[0][0])
    label = np.array(label)
    a = np.column_stack((a[:, :256], label.T))
    a = a.astype("float")

    if save_flag == 1:
        np.save("./dataSet/semeion_finally.npy", a)  # 保存为.npy格式


if __name__ == '__main__':
    pass
    # set_working_dir()
    import os
    #
    # # 获取当前文件的目录
    #
    d = get_data("wdbc.npy")
    wdbc_preprocessing(d, save_flag=1)
    #
    # # train_X, train_y, test_X, test_y = load_data("clean2_finally.npy")
    # # X, y = test_X, test_y
    # # discretization_caimcaim(X, y, file_name="clean2", save_flag=1)
