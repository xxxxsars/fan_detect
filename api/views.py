# -*- coding: utf-8 -*-
'''
Created on 2021/2/24

Author Andy Huang
'''
from rest_framework.decorators import api_view
from django.http.response import JsonResponse
from .handler import *
from fan_detect import settings
import socket
import requests

@api_view(["POST"])
def sfc_checkin(request):

    check_keys = ["work_order","sn","user_id"]

    for key in check_keys:
        if key not in request.data:
            return JsonResponse({"message": f"The '{key}' not existed"}, status=417)

    try:
        work_order = request.data.get("work_order")
        sn = request.data.get("sn")
        user_id = request.data.get("user_id")
        checkin(work_order,sn,user_id)
    except Exception as e:
        return JsonResponse({"message": f"Check In had error:{e}"}, status=417)

    return JsonResponse({"message": "check in successfully."}, status=200)


@api_view(["POST"])
def sfc_checkout(request):

    check_keys = ["work_order","sn","test_result"]
    for key in check_keys:
        if key not in request.data:
            return JsonResponse({"message": f"The '{key}' not existed"}, status=417)


    try:
        work_order = request.data.get("work_order")
        sn = request.data.get("sn")
        test_result = request.data.get("test_result")
        checkout(work_order, sn, test_result)
    except Exception as e:
        return JsonResponse({"message": f"Check Out had error:{e}"}, status=417)
    return JsonResponse({"message": "check out successfully."}, status=200)

@api_view(["GET"])
def sfc_status(request):
    try:
        save = get_ini()["SFC_SETTING"]["save"]
    except Exception as e:
        return JsonResponse({"message": f"Get sfc status error:{e}"}, status=417)

    if save !="True":
        return JsonResponse({"status": False}, status=200)
    else:
        return JsonResponse({"status": True}, status=200)

@api_view(["GET"])
def socket_status(request):
    try:
        g = get_socket_status()
    except Exception as e:
        return JsonResponse({"status": f"Initial"}, status=417)

    status = g["status"]
    if status == "Finish":
        return JsonResponse({"status":status }, status=200)
    else:
        return JsonResponse({"status": status}, status=417)

@api_view(["POST"])
def predict_image(request):
    if request.method == "POST":
        img = request.data.get("image")
        sn = request.data.get("sn")
        work_order = request.data.get("work_order")

        try:
            data = predict(img, work_order, sn)
            predict_result = data["predict_class"]
            base64_img = data["img"]
        except Exception as e:
            return JsonResponse({"message": f"Predict image had error:{e}"}, status=417)

        if predict_result  == "PASS" :
            message = "Fan detection passed."
        elif predict_result =="FAIL":
            message = "Fan detection failed."
        else:
            message = "Detection failed , please try again."

        save_condition = ["PASS","FAIL"]
        try:
            config = get_ini()["FILE_SETTING"]
            if (config["save"]) == "True" and predict_result in save_condition:
                print(predict_result)
                path = handle_path(config["path"], work_order, predict_result,f"{sn}.jpg")
                print(path)
                save_base64(base64_img,path)
        except Exception as e:
            return JsonResponse({"message": f"Saving result image had error:{e}"}, status=417)


        return JsonResponse({"message":message,"image":base64_img,"predict":predict_result}, status=200)



def predict(image:str, work_order:str, sn:str)->str:

    data =  {'work_order': work_order, 'sn': sn,'image': image}

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((settings.SOCKET_HOST, settings.SOCKET_PORT))
    client.sendall(str(data).encode())

    data =  recvall(client)

    had_error = re.search(r"Error : (.+)", data["predict_class"])
    client.close()

    if had_error:
        raise Exception(had_error.group(1))
    else:
        return data

