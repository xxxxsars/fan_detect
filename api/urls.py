# -*- coding: utf-8 -*-
'''
Created on 2021/2/24

Author Andy Huang
'''
from django.conf import settings
from django.conf.urls import url, include
from api.views import *


urlpatterns = [
    url("^predict_image/$", predict_image),
]

