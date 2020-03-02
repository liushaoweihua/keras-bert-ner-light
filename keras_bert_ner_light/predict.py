# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: predict.py.py
@Time: 2020/3/2 5:12 PM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import codecs
import pickle
from keras.models import load_model
from keras_contrib.layers import CRF
from .utils.decoder import Viterbi
from .utils.metrics import CrfAcc, CrfLoss
from .utils.featurizer import Featurizer


def build_trained_model(args):
    """模型加载流程
    """
    # 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_map if args.device_map != "cpu" else ""
    # 处理流程
    with codecs.open(os.path.join(args.file_path, "tag_to_id.pkl"), "rb") as f:
        tag_to_id = pickle.load(f)
    with codecs.open(os.path.join(args.file_path, "id_to_tag.pkl"), "rb") as f:
        id_to_tag = pickle.load(f)
    crf_accuracy = CrfAcc(tag_to_id, args.tag_padding).crf_accuracy
    crf_loss = CrfLoss(tag_to_id, args.tag_padding).crf_loss
    custom_objects = {
        "CRF": CRF,
        "crf_accuracy": crf_accuracy,
        "crf_loss": crf_loss}
    model = load_model(args.model_path, custom_objects=custom_objects)
    model._make_predict_function()
    viterbi_decoder = Viterbi(model, len(id_to_tag))

    return id_to_tag, viterbi_decoder


def get_model_inputs(url, port, timeout, texts):
    """获取模型的预测输入
    """
    featurizer = Featurizer(url, port, timeout)
    if isinstance(texts, str):
        texts = [texts]

    return featurizer.parse(texts)