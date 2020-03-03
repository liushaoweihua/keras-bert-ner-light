# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: featurizer.py
@Time: 2020/3/2 3:39 PM
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import requests
import numpy as np


class Featurizer:

    def __init__(self, url, port, timeout):
        self.url = "%s:%s/encode" % (url, port)
        self.timeout = timeout

    def parse(self, data):
        if isinstance(data, str):
            data = [data]
        if isinstance(data, np.ndarray):
            data = data.tolist()
        timeout = self.timeout * len(data)
        r = requests.post(
            self.url,
            json={
                "id": 1225,
                "texts": data,
                "is_tokenized": False
            },
            timeout=timeout)

        return np.array(json.loads(r.text).get("result"))