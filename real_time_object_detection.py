# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:01:50 2020

@author: by_to
"""
#Ввевсти в терминал:
#python "C:\Users\by_to\faceRec\real_time_object_detection.py" --prototxt "C:\Users\by_to\faceRec\MobileNet_deploy.prototxt" --model "C:\Users\by_to\faceRec\MobileNet_deploy.caffemodel"

# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# парс
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Инициализируем метку и генерируем цвет ограничительного прямоугольника
CLASSES = ["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# Загружаем модель с диска
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Камера
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Цикл по кадрам видеопотока
while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	
	net.setInput(blob)
	detections = net.forward()

	# Цикл на обнаружение
	for i in np.arange(0, detections.shape[2]):
		# Достоверность, связанная с прогнозом
		confidence = detections[0, 0, i, 2]

		# Фильтр слабых обнаружений
		if confidence > args["confidence"]:
			# (x, y) -координаты ограничительной рамки объекта
			idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Отобразить прогноз на кадре
			label = "{}: {:.2f}%".format(CLASSES[0],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[0], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# q- завершение
	if key == ord("q"):
		break

	
	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
