# -*- coding: utf-8 -*-
'''
Created on 2021/2/24

Author Andy Huang
'''
import os
import datetime
import shutil
import uuid
from django.shortcuts import render
from django.http import HttpResponse, Http404, JsonResponse
from django.contrib.sessions.models import Session
from django.utils import timezone

from api.handler import *
from fan_detect.settings import *

def index(request):


    return render(request, 'index.html', locals())