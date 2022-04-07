#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: wangyongpeng
# Mail: 1690992651@qq.com
# Created Time:  2022-03-15

# 1690992651@qq.com
# username: wangyongpeng
# pwd : xedpix-zopby1-jYhdin

#############################################

from setuptools import setup, find_packages            #这个包没有的可以pip一下

setup(
    name="ZhuanZhuanBktUtil",      #这里是pip项目发布的名称
    version="3.0.0",  #版本号，数值大的会优先被pip
    keywords=("pip", "wangyongpeng","tree"),
    description="转转分桶工具。",
    long_description="转转分桶工具",
    license="BSD Licence",

    url="https://github.com/HeiBoWang/ZZBktUtils",     # 项目相关文件地址，一般是github
    author="wangyongpeng",
    author_email="wangyongpeng@zhuanzhuan.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any"
)


