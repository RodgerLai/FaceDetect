import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
from PIL import Image
import os
import urllib
import urllib.request
import base64
from bson import json_util

BAIDU_AI_ACCESS_TOKEN = ''

def getBaiduAIToken():
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=jQ0uW4jkHwZwek1zUsdqt1lp&client_secret=a25wwGAdVnWV6VGjV8sxRVaYoUqyxaRW'
    request = urllib.request.Request(host)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = urllib.request.urlopen(request)
    content = response.read()
    if (content):
        print("获取百度AI token:")
        print(content)
        result = json_util.loads(content)
        return result["access_token"]


def detectFaceByBaiduAI(token, image):
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"
    ## face_field 去除 landmark landmark72 landmark150
    data = {"image": image, "image_type": "BASE64",
            "face_field": "age,beauty,expression,face_shape,gender,glasses,race,quality,eye_status,emotion,face_type"}
    request_url = request_url + "?access_token=" + token
    request = urllib.request.Request(url=request_url, data=urllib.parse.urlencode(data).encode(encoding='UTF8'))
    request.add_header('Content-Type', 'application/json')
    response = urllib.request.urlopen(request)
    content = response.read()
    if content:
        print(' ')
        print("脸部检测结果：")
        print(content)
        res = json_util.loads(content)
        if (res['error_msg'] == 'SUCCESS'):
            print("人脸识别成功...")
            result = res['result']
            face_list = result['face_list']
            index = 1
            for face in face_list:
                print("识别第【" + str(index) + "】个人=>", end=' ')
                print("性别：" + decodeGender(face['gender']['type']) + " 可信度：" + str(face['emotion']['probability']), end=' ')
                print(";  表情：" + decodeEmotion(face['emotion']['type']) + " 可信度：" + str(face['emotion']['probability']), end=' ')
                print(' ')
        else:
            print("人脸识别失败：" + res['error_msg'])


# angry:愤怒 disgust:厌恶 fear:恐惧 happy:高兴
# sad:伤心 surprise:惊讶 neutral:无情绪
def decodeEmotion(value):
    text = '无情绪'
    if value == 'angry':
        text = '愤怒'
    if value == 'disgust':
        text = '厌恶'
    if value == 'fear':
        text = '恐惧'
    if value == 'happy':
        text = '高兴'
    if value == 'sad':
        text = '伤心'
    if value == 'surprise':
        text = '惊讶'
    if value == 'neutral':
        text = '无情绪'

    return text


def decodeGender(value):
    if value == 'male':
        return '男性'
    if value == 'female':
        return '女性'

    return '未知'


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    sleep(5)
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if ret:
        # 制作缩略图
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((500, 300))
        if not os.path.exists('./picture'):
            os.mkdir('./picture')
        imageFilePath = "./picture/face" + dt.datetime.now().strftime('%Y%m%d%H%M%S') + ".jpg";
        image.save(imageFilePath, format='jpeg')
        imageBase64 = ''
        with open(imageFilePath, 'rb') as imageFile:
            base64_data = base64.b64encode(imageFile.read())
            imageBase64 = base64_data.decode()
        del image
        # del frame, ret, image
        # break
        if BAIDU_AI_ACCESS_TOKEN == '':
            BAIDU_AI_ACCESS_TOKEN = getBaiduAIToken()
        detectFaceByBaiduAI(BAIDU_AI_ACCESS_TOKEN, imageBase64)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
