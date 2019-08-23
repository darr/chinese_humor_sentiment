#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2019-08-20 18:09
# Modified date : 2019-08-23 15:11
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from humor_category_classifier_contrast import HumorCategoryClassify
#from humor_category_train import HumorCategoryClassify
from humor_degree_train import HumorDegreeClassify
from transfer_data import TransferData
from yinyu_senti_train import YinyuSentimentClassify
from yinyu_type_train import YinyuTypeClassify

def run_humor_degree():
    print("humor degree:")
    ner = HumorDegreeClassify()
    ner.train_model()

def run_humor_category():
    print("humor category:")
    handler = HumorCategoryClassify()
    handler.test_model()

def run_transfer_data():
    print("transfer data:")
    handler = TransferData()
    handler.transfer_category()
    handler.transfer_degree()
    handler.transfer_senti()
    handler.transfer_verb()

def run_yinyu_senti():
    print("yinyu senti classify:")
    ner = YinyuSentimentClassify()
    ner.train_model()

def run_yinyu_type():
    print("yinyu type classify:")
    ner = YinyuTypeClassify()
    ner.train_model()

def run():
    #run_humor_category()
    run_humor_degree()
    #run_transfer_data()
    #run_yinyu_senti()
    #run_yinyu_type()

run()
