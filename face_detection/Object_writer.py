import dlib
from skimage import io
from scipy.spatial import distance
import cv2

set_frame_path = 'set/Frame.jpg'

def write_obj(frame, detector, sp, facerec):
    face_id = input("Обозначьте имя обьекта : ")
    cv2.imwrite(set_frame_path, frame)
    img = io.imread(set_frame_path)
    dets = detector(img, 1)
    for _, d in enumerate(dets): # Цикл для перебора обнаруженных лиц
        shape = sp(img, d)
    face_descriptor1 = facerec.compute_face_descriptor(img, shape)  # Вычисление дескрипторов
    file = open('Face.txt', 'a')
    file.write(str(face_descriptor1).replace('\n', ' ') + ' ' + face_id + '\n')  # Записываем файл
    cv2.imwrite("set/" + face_id + '.jpg', frame)  # Сохраняем изображение
    file.close()