import dlib
from skimage import io
from scipy.spatial import distance
import cv2

frame_frame_path = 'Frame/Frame.jpg'

def predict_face(item, flag, count, frame, detector, sp, facerec, rect, face_file_len):
    x, y, w, h = rect
    predict = []
    item = item.split(' ')
    name = str([line.rstrip() for line in item[128:129]]).replace('[', '').replace(']', '').replace("'", "")
    item = item[:-1]
    for i in item:
        predict.append(float(i))
    cv2.imwrite(frame_frame_path, frame)
    img = io.imread(frame_frame_path)  # Прочитать кадр
    dets_webcam = detector(img, 1)
    if len(dets_webcam) < 1:
        return False
    for _, d in enumerate(dets_webcam):
        shape = sp(img, d)
    face_descriptor2 = facerec.compute_face_descriptor(img, shape)  # Вычисление дескрипторов
    a = distance.euclidean(predict, face_descriptor2)  # Сравнение дескрипторов
    print(a)
    if count >= face_file_len:
        return True
    if a < 0.6:
        print("Hello " + name)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Рисование прямоугольника
        cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_DUPLEX, 1,
                    (0, 255, 255), 2)  # Нарисовать текст
        cv2.putText(frame, str(110 - int((100 * a))) + '%', (x + 5, y + h - 5), cv2.FONT_HERSHEY_DUPLEX, 1,
                    (0, 255, 0), 1)
        return True
    return False