# -*- coding: utf-8 -*-
'''
Created on 2021/2/24

Author Andy Huang
'''
import base64
import os
import re
from PIL import Image
import io
import fcntl
import json

from fan_detect.settings import SOCKET_STATUS
from fan_detect import settings

def handle_path(root_path:str, *arg: str) -> str:
    for path in arg:
        root_path = os.path.join(root_path, path)

    if re.search(r"\.", os.path.basename(root_path)):
        check_path = os.path.dirname(root_path)

    else:
        check_path = root_path

    if not os.path.exists(check_path):
        os.makedirs(check_path)

    return root_path


def list_cfile(path: str) -> list:
    files = []
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            file_path = os.path.abspath(os.path.join(dirpath, f))
            files.append((os.stat(file_path).st_ctime, file_path))
    return  files


def list_file(path: str) -> list:
    files = list_cfile(path)
    # sorted file by create time
    sorted_files = sorted(files, key=lambda f: f[0])
    return [f[1] for f in sorted_files]

def save_base64(img_str:str,out_file:str,resize:bool=False):
    image = base64.b64decode(img_str)
    imagePath = (out_file)
    img = Image.open(io.BytesIO(image))

    if resize:
        (w, h) = img.size
        img = img.resize((int(w/5), int(h/5)))
    img.save(imagePath, 'jpeg')



def clean_file(sn:str):
    path = (handle_path(settings.MEDIA_ROOT,"image"))
    for p in list_file(path):
        file_name = os.path.split(p)[-1]
        if file_name == f"{sn}.jpg":
            try:
                os.remove(p)
            except Exception as e:
                print(f"Error: {e} , in the {p}")



def get_socket_status():
    with open(settings.SOCKET_STATUS,"r") as fin:
        fcntl.flock(fin, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return json.load(fin)


def update_status(status:str):
    fin =  open(SOCKET_STATUS,'w')

    fcntl.flock(fin, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = {"status":status}
    json.dump(result,fin)

    fcntl.flock(fin, fcntl.LOCK_UN)
    fin.close()