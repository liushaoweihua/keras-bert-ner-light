# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: feature_server.py
@Time: 2020/3/2 2:50 PM
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from bert_encode_server.server import BertServer
from bert_encode_server.server.helper import  get_run_args


if __name__ == "__main__":
    with BertServer(get_run_args()) as server:
        server.join()
