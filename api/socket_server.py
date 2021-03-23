# -*- coding: utf-8 -*-
'''
Created on 2021/2/25

Author Andy Huang
'''

import socket
import logging
import cv2
import tensorflow as tf
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)



from api.object_detection.utils import ops as utils_ops
from api.object_detection.utils import label_map_util
from api.object_detection.utils import visualization_utils as vis_util

from handler import *
from fan_detect.settings import MEDIA_ROOT,SOCKET_PORT,SOCKET_HOST,LOG_PATH



logging.basicConfig(filename=LOG_PATH,
                    filemode='a',
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.ERROR)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)


    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def load_model(model_dir):
    model = tf.saved_model.load(model_dir)
    return model

def load_image(image_path:str)->np.array:
    img = Image.open(image_path)
    image_np = np.array(img)
    return  image_np

def img_to_np(img:str)->np.array:
    img_data = base64.b64decode(img)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return  img_np

def np_to_base64(img_np)->str:
    img = Image.fromarray(img_np)
    output_buffer = io.BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)

    return base64_str.decode()

def load_label(label_path):
    return label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)

def show_inference(model,category_index, image_np):

    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)


    # Visualization of the results of a detection.
    _,predict_class = vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=4,
      skip_scores = True)

    result = {"predict_class":predict_class ,"image_np":  image_np}

    return  result

def test_predict(model,label,img_path):
    img_np = load_image(img_path)
    category_index = label
    predict_result = show_inference(model,category_index,img_np)

    predict_class = ",".join(predict_result["predict_class"])

    return  predict_class


def predict_image(model,label,img_np,work_order,sn)->str:

    category_index = label

    predict_result = show_inference(model,category_index,img_np)

    result_img = predict_result["image_np"]

    img_str = np_to_base64(result_img)

    if not predict_result["predict_class"]:
        predict_class = "Not Detect"
    elif len(predict_result["predict_class"]) >1:
        predict_class = "Detect Failed"
    else:
        predict_class = ",".join(predict_result["predict_class"])


    return  str({"predict_class":predict_class,"img":img_str})


class SOCKERT_SERVER:
    def __init__(self):
        self.host = SOCKET_HOST
        self.prot = SOCKET_PORT
        self.model = load_model(handle_path(MEDIA_ROOT, "saved_model"))
        self.label =  load_label(  handle_path(MEDIA_ROOT ,'label_map.pbtxt'))


    def start(self):
        update_status("Initial")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.prot))
        server.listen(1)

        test_class = test_predict(self.model, self.label, handle_path(MEDIA_ROOT,"test.jpg"))
        assert test_class =="PASS","Initial predict image had error."

        time = (datetime.datetime.now()).strftime("%d/%b/%Y %H:%H:%S")
        print( f"[{time}] Model load successfully...")
        update_status("Finish")


        while True:
            conn, addr = server.accept()
            #Runtime error will show on web page
            try:
                recv_data = recvall(conn)
                img_np = (img_to_np(recv_data['image']))
                work_order = recv_data["work_order"]
                sn = recv_data["sn"]
                predict_class = predict_image(self.model, self.label, img_np, work_order, sn)
                conn.sendall(predict_class.encode())
            except Exception as e:
                err  = f"Error : {e}"
                conn.sendall(err.encode())
            finally:
                conn.close()

if __name__ =="__main__":
    #limit gpu memory usage szie.
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        # Restrict TensorFlow to only allocate 3GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
            logging.error(e)

    #Initial socket server error will record to log.
    s = SOCKERT_SERVER()
    try:
        s.start()
    except Exception as e:
        print(e)
        logging.error(e)



