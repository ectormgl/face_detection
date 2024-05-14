import cv2
import os 
from PIL import Image
import numpy as np
pasta = r'./new_images'
train_path= os.path.join(pasta, 'train')
test_path= os.path.join(pasta, 'test')
def ImageData():
    img_data=[]
    ids_num=[]
    for i in os.listdir(train_path):
        ppp= os.path.join(train_path, i)
        temp_img= Image.open(ppp)
        temp_img= np.array(temp_img)
        img_data.append(temp_img)
        id = int(os.path.split(i)[1].split('.')[0].replace('subject', ''))
        ids_num.append(id)
    return ids_num, img_data 
        
ids_numbers, images= ImageData()

ids_numpy= np.zeros_like(ids_numbers)
cont=0
for i in ids_numbers:
    ids_numpy[cont]= i
    cont+=1


ar= cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=12)
ar.train(images, ids_numpy)
ar.write('lbph_algorithm.yml')


