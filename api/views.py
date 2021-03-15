# -*- coding: utf-8 -*-
'''
Created on 2021/2/24

Author Andy Huang
'''
from rest_framework.decorators import  api_view
from django.http.response import JsonResponse
from .handler import *
from fan_detect import settings
import socket



@api_view(["POST"])
def predict_image(request):
    if request.method == "POST":
        img = request.data.get("image")
        sn = request.data.get("sn")


        try:
            root_path = handle_path(settings.MEDIA_ROOT, "image")
        except Exception as e:
            return JsonResponse({"message": f"Get the predict image had error:{e}"}, status=417)

        try:
            # Remove duplicate files
            clean_file(sn)
            img_path = handle_path(root_path,"source",f"{sn}.jpg")
            save_base64(img, img_path)

        except Exception as e:
            return JsonResponse({"message": f"Handle a sample image had error:{e}"}, status=417)

        try:
             predict_result = predict(sn)
        except Exception as e:
            return JsonResponse({"message": f"Predict a image had error:{e}"}, status=417)



        if predict_result  == "PASS" :
            message = "Fan detection passed."
        elif predict_result =="FAIL":
            message = "Fan detection failed."
        else:
            message = "Detection failed , please try again."

        out_img_path = handle_path(root_path,"result",predict_result,f"{sn}.jpg")

        with open(out_img_path, "rb") as image_file:
            encoded_string = (base64.b64encode(image_file.read())).decode('utf-8')


        return JsonResponse({"message":message,"image":encoded_string,"predict":predict_result}, status=200)



def predict(sn:str)->str:
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((settings.SOCKET_HOST, settings.SOCKET_PORT))
    client.sendall(sn.encode())

    predict_class = str(client.recv(1024), encoding='utf-8')
    had_error = re.search(r"Error : (.+)", predict_class)
    client.close()

    if had_error:
        raise Exception(had_error.group(1))
    else:
        return predict_class

