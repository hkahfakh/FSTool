#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
距离相关系数是为了克服Pearson相关系数的弱点而生的。
在x和x^2这个例子中，即便Pearson相关系数是0，我们也不能断定这两个变量是独立的（有可能是非线性相关）；
但如果距离相关系数是0，那么我们就可以说这两个变量是独立的。
"""
from .relief import Relief
from .reliefF import ReliefF
