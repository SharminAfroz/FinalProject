import numpy as np
import os
import cv2


class FaceCrop:
    __slots__ = ['datadir', 'save_path']

    def __int__(self, datadir: str, save_path: str):
        self.datadir = datadir
        self.save_path = save_path

    def check_dir_existance(self, dataset_directories:list) -> None:
        base_dir_exists = os.path.exists(self.save_path)
        if not base_dir_exists:
            os.mkdir(self.save_path)
        for item in dataset_directories:
            new_path = f'{self.save_path}/{item}'
            is_new_path_exists = os.path.exists(new_path)
            if not is_new_path_exists:
                os.mkdir(new_path)
        print('Dataset directories created.....!!!')

    def crop_face(self)-> None:
        items = sorted(os.listdir(self.datadir))
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.check_dir_existance(items)
        for item in range(len(items)):
            path = os.path.join(self.datadir, items[item])
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path,img))
                    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    for (x,y,w,h) in faces:
                        face_crop = img_arr[y:y+h, x:x+w]
                        save_to = f'{self.save_path}/{items[item]}/{img}'
                        print(f'saving to {save_to}')
                        cv2.imwrite(save_to, face_crop)
                except:
                    print(f'Unable to save {img} !!!!!')
        print('saving done ............')



fc = FaceCrop()
fc.datadir='/home/sharmin/Downloads/old/masked face Detect/dataset'
fc.save_path='/home/sharmin/Videos/datasets'

fc.crop_face()
