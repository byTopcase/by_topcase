# Подключаемые библиотеки
import dlib
from skimage import io
from scipy.spatial import distance
import cv2
from Face_predictor import predict_face
from Object_writer import write_obj


# python Face_detection.py


# Модули для работы с видео потоком и обученные модели для распознавания лиц
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

# Запрос на формирование БД
AddData = int(input("Записать данные? (0-Да) (1-Нет): "))
if AddData == 0:
    choose = 'Choose'
    flagg = True
    while choose != 'Нет':
        ret, frame = cap.read()  # Получаем кадр
        faces = face_detector.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            w = w + 8
            h = h + 28
            cv2.putText(frame, "Continue ?", (x + 5, y - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)  # Рисование текста
            cv2.imshow("Frame", frame)  # Показать кадр
            cv2.waitKey()
            choose = int(input("Добавить в базу данных?(0-Да)-(1-Нет) : "))
            if choose == 0:
                write_obj(frame[y:y + h, x:x + w], detector, sp, facerec)
                flagg = False
        if flagg:
            print("Обьект для добавления не обнаружен!")
        choose = input("Продолжить добавление?(Нет) (Да) : ")

face_file = open('Face.txt').readlines()  # Считывание БД

# Распознавание лиц:
while True:
    ret, frame = cap.read()  # Получаем кадр
    faces = face_detector.detectMultiScale(frame, 1.3, 5)  # Обнаружение лиц
    for rect in faces:  # Цикл сопоставления дескрипторов из БД и обнаруженных лиц
        x, y, w, h = rect
        flag = False
        count = 0
        while not flag:
            for item in face_file: # Считывание из БД каждого лица
                count += 1
                flag = predict_face(
                    item, flag, count,
                    frame[y:y + h, x:x + w], detector, sp, facerec,
                    rect, len(face_file))
                if flag:
                    break
    cv2.imshow('Faces', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):  # Обнаружение каждую секунду
        break

cap.release()
cv2.destroyAllWindows()