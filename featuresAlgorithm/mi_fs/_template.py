#!/usr/bin/env python
# -*- coding: utf-8 -*-
# New Algorithm Template
from featuresAlgorithm.base import MIFSBase


class Template(MIFSBase):

    def __init__(self, F, C, m=3):
        super(Template, self).__init__(F, C, m)

    def feature_selection(self):

        for _ in range(self.m):
            pass


if __name__ == '__main__':
    pass
