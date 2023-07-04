#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil

source_path = './'  # 替换成你的源文件夹路径
destination_path = './dpath'  # 替换成你的目标文件夹路径


def delete_files_in_subfolders(path):
    for root, directories, files in os.walk(path):
        if root != path:  # 排除根目录
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)


def move_files_to_folder(source_path, destination_path):
    os.makedirs(destination_path, exist_ok=True)  # 创建目标文件夹

    for root, directories, files in os.walk(source_path):
        if root != source_path and root != destination_path:  # 排除根目录
            for file in files:
                file_path = os.path.join(root, file)

                folder_name = os.path.basename(root)
                new_file_name = folder_name + '_' + file
                new_file_path = os.path.join(destination_path, new_file_name)
                shutil.move(file_path, new_file_path)


if __name__ == '__main__':
    # delete_files_in_subfolders(source_path)
    move_files_to_folder(source_path, destination_path)
