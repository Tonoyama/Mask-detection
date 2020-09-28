# -*- coding:utf-8 -*-
import paddlehub as hub
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os

module = hub.Module(name="pyramidbox_lite_server_mask", version='1.1.0')

black_img = cv2.imread("./white.png", 0)


def paint_japanese(im, japanese, position, fontsize, color_bgr):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(
        'TakaoGothic.ttf', size=35, encoding="utf-8")
    color = color_bgr[::-1]
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, japanese, font=font, fill=color)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


result_path = './result'
if not os.path.exists(result_path):
    os.mkdir(result_path)


width = 1280
height = 720
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'vp90')
writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

maskIndex = 0
index = 0
data = []


'''
        if label == 'NO MASK':
            color = (0, 0, 255)
            label_cn = "マスクを\n着用して下さい"

'''

capture = cv2.VideoCapture(-1)
while True:
    frameData = {}
    ret, frame = capture.read(-1)
    if ret == False:
        break

    frame_copy = frame.copy()
    input_dict = {"data": [frame]}
    results = module.face_detection(data=input_dict)

    maskFrameDatas = []
    for result in results:
        label = result['data']['label']
        confidence_origin = result['data']['confidence']
        confidence = round(confidence_origin, 2)
        confidence_desc = str(confidence)

        top, right, bottom, left = int(result['data']['top']), int(
            result['data']['right']), int(result['data']['bottom']), int(
                result['data']['left'])

        img_name = "avatar_%d.png" % (maskIndex)
        path = "./result/" + img_name
        image = frame[top - 10:bottom + 10, left - 10:right + 10]

        maskIndex += 1

        maskFrameData = {}
        maskFrameData['top'] = top
        maskFrameData['right'] = right
        maskFrameData['bottom'] = bottom
        maskFrameData['left'] = left
        maskFrameData['confidence'] = float(confidence_origin)
        maskFrameData['label'] = label
        maskFrameData['img'] = img_name

        maskFrameDatas.append(maskFrameData)

        color = (0, 255, 0)
        label_cn = "OK"
        cv2.namedWindow("Put on a Mask!", cv2.WINDOW_NORMAL)
        cv2.imshow("Put on a Mask!", black_img)
        if label == 'NO MASK':
            color = (0, 0, 255)
            label_cn = "マスクを\n着用して下さい"
            try:
                cv2.namedWindow("Put on a Mask!", cv2.WINDOW_NORMAL)
                cv2.imshow("Put on a Mask!", image)
            except:
                pass

        cv2.rectangle(frame_copy, (left, top), (right, bottom), color, 3)
        #cv2.putText(frame_copy, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        origin_point = (left, top - 70)
        frame_copy = paint_japanese(
            frame_copy, label_cn, origin_point, 24, color)
        warning_word = "マスクを付けましょう！"
        frame_copy = paint_japanese(
            frame_copy, warning_word, (15, 20), 24, color)

    writer.write(frame_copy)

    cv2.namedWindow("Mask detection", cv2.WINDOW_NORMAL)

    cv2.imshow('Mask detection', frame_copy)

    frameData['frame'] = index
    # frameData['seconds'] = int(index/fps)
    frameData['data'] = maskFrameDatas

    data.append(frameData)
    print(json.dumps(frameData))

    index += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open("./result/2-mask_detection.json", "w") as f:
    json.dump(data, f)

writer.release()

cv2.destroyAllWindows()
