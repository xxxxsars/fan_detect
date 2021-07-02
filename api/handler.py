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
import json
import pytz
import datetime
import requests
import platform
import configparser
import urllib.parse as urlparse
from urllib.parse import urlencode
import numpy as np
from fan_detect.settings import SOCKET_STATUS, CONFIG_PATH
from fan_detect import settings


def handle_path(root_path: str, *arg: str) -> str:
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
    return files


def list_file(path: str) -> list:
    files = list_cfile(path)
    # sorted file by create time
    sorted_files = sorted(files, key=lambda f: f[0])
    return [f[1] for f in sorted_files]


def save_base64(img_str: str, out_file: str, resize: bool = False):
    image = base64.b64decode(img_str)
    imagePath = (out_file)

    byteImgIO =  io.BytesIO(image)
    byteImgIO.seek(0)
    img = Image.open(byteImgIO)

    if resize:
        (w, h) = img.size
        img = img.resize((int(w / 5), int(h / 5)))
    img.save(imagePath, 'jpeg')

def clean_file(sn: str):
    path = (handle_path(settings.MEDIA_ROOT, "image"))
    for p in list_file(path):
        file_name = os.path.split(p)[-1]
        if file_name == f"{sn}.jpg":
            try:
                os.remove(p)
            except Exception as e:
                print(f"Error: {e} , in the {p}")


def get_socket_status():
    with open(settings.SOCKET_STATUS, "r") as fin:
        # fcntl.flock(fin, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return json.load(fin)


def update_status(status: str):
    fin = open(SOCKET_STATUS, 'w')

    # fcntl.flock(fin, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = {"status": status}
    json.dump(result, fin)

    # fcntl.flock(fin, fcntl.LOCK_UN)
    fin.close()


def merge_url(url: str, params: dict) -> str:
    url_parts = list(urlparse.urlparse(url))
    query = dict(urlparse.parse_qsl(url_parts[4]))

    query.update(params)
    url_parts[4] = urlencode(query)

    return urlparse.urlunparse(url_parts)


def get_ini() -> dict:
    conf = configparser.ConfigParser()
    conf.read(CONFIG_PATH, encoding="utf-8")
    #conf.read("config.ini", encoding="utf-8")

    result = {}
    for section in (conf.sections()):
        data = {}
        for c in conf.items(section):
            data[c[0]] = c[1]
        result[section] = data

    return result


def get_time() -> str:
    tz = pytz.timezone('Asia/Jakarta')
    time = datetime.datetime.now(tz)

    return time.strftime('%Y%m%d%H%M%S')


def get_url(api_type: str, params: dict) -> str:
    config = get_ini()["SFC_SETTING"]
    if config["save"] =="True":
        ip = config['ip']
        port = config['port']
        suffix = config[api_type]
        url = merge_url(f"http://{ip}:{port}/{suffix}", params)
        return url

    else:
        raise Exception("Not allow access the sfc api,please modify your 'config.ini' file.")


def get_model_name(work_order: str) -> str:
    params = {"work_order": work_order}
    url = get_url("getpn", params)

    retry = True
    retry_count = 0
    resp_json = ""

    while retry:
        if retry_count < 10:
            try:
                resp = requests.get(url, timeout=0.1)
                retry = False

                resp_json = resp.json()
            except Exception as e:
                print(e)
                retry_count += 1
        else:
            raise Exception(f"Connect '{url}' had error.")
    if "PART_NO" in resp_json:

        return (resp_json["PART_NO"])[:7]
    else:
        raise Exception("Get model name api had error.")


def checkin(work_order: str, sn: str,user_id: str):
    model_name = get_model_name(work_order=work_order)

    time = get_time()

    if platform.system() =="Windows":
        machine_id = os.environ['COMPUTERNAME']
    else:
        machine_id = platform.uname()[1]
    params = {'sn': sn, 'sn_type': 'SN', 'model_name': model_name, 'user_id': user_id,
              'station_name': 'ASSY_AOI_FT1', 'machine_id': machine_id, 'work_order': work_order,
              'start_time': time}
    url = get_url("checkin", params)
    resp = requests.post(url, data=params,timeout=1)
    resp_json = resp.json()["SECTION_APIINFO"]

    if resp_json["API_RESULT"] != "OK":
        raise Exception(resp_json["API_ERROR_MESSAGE"])


def checkout(work_order: str, sn: str,test_result: str):
    time = get_time()
    params = {'sn': sn, 'sn_type': 'SN', 'station_name': 'ASSY_AOI_FT1', 'work_order': work_order,
              'end_time': time, 'test_result': test_result}

    url = get_url("checkout", params)
    resp = requests.post(url, data=params,timeout=1)
    resp_json = resp.json()["SECTION_APIINFO"]

    if resp_json["API_RESULT"] != "OK":
        raise Exception(resp_json["API_ERROR_MESSAGE"])

def recvall(sock):
    BUFF_SIZE = 4096 # 4 KiB
    data = b''
    #sock.setblocking(0)
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break

    print(len(data))

    json_data = json.loads((data.decode("utf-8").replace("'", '"')))
    # print("json_data",json_data)
    return json_data


