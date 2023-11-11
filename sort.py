import os, cv2
import os.path
import numpy as np
import shutil
from glob import glob

folder_path = "./"

images = glob('/home/shussain/Downloads/Car_Dataset/Sort_img/car/*.jpg')
print(len(images))
excluded = []

for image in images:
    #print(image)
    try:
        show = cv2.imread(image)
        img = cv2.resize(show, (720, 600))
        cv2.imshow(image, img)
        k = cv2.waitKey(0)
        if k == ord('w'):
            #print('white')
            folder_name = 'white'
            os.makedirs(folder_name, exist_ok = True)
            shutil.move(image, folder_name+'/'+image.split('/')[-1])

        elif k == ord('b'):
            #print('black')
            folder_name = 'black'
            os.makedirs(folder_name, exist_ok = True)
            shutil.move(image, folder_name+'/'+image.split('/')[-1])
            
        elif k == ord('l'):
            #print('blue')
            folder_name = 'blue'f
            os.makedirs(folder_name, exist_ok = True)
            shutil.move(image, folder_name+'/'+image.split('/')[-1])h
            
        elif k == ord('g'):
            #print('gray')
            folder_name = 'gray'
            os.makedirs(folder_name, exist_ok = True)
            shutil.move(image, folder_name+'/'+image.split('/')[-1])
            
        elif k == ord('r'):
            #print('red')
            folder_name = 'red'
            os.makedirs(folder_name, exist_ok = True)
            shutil.move(image, folder_name+'/'+image.split('/')[-1])
            
        elif k == ord('y'):
            #print('yellow')
            folder_name = 'yellow'
            os.makedirs(folder_name, exist_ok = True)
            shutil.move(image, folder_name+'/'+image.split('/')[-1])
            
        elif k == ord('f'):
            #print('brown')
            folder_name = 'brown'
            os.makedirs(folder_name, exist_ok = True)
            shutil.move(image, folder_name+'/'+image.split('/')[-1])
            
        elif k == ord('s'):
            #print('silver')
            folder_name = 'silver'
            os.makedirs(folder_name, exist_ok = True)
            shutil.move(image, folder_name+'/'+image.split('/')[-1])
            
        elif k == ord('p'):
            #print('pink')
            folder_name = 'pink'
            os.makedirs(folder_name, exist_ok = True)
            shutil.move(image, folder_name+'/'+image.split('/')[-1])
            
        elif k == ord('h'):
            #print('green')
            folder_name = 'green'
            os.makedirs(folder_name, exist_ok = True)
            shutil.move(image, folder_name+'/'+image.split('/')[-1])
            
        elif k == ord('u'):
            #print('u')
            folder_name = 'unknown'
            os.makedirs(folder_name, exist_ok = True)
            shutil.move(image, folder_name+'/'+image.split('/')[-1])
            
        elif k == ord('m'):
            #print('multi')
            folder_name = 'multi'
            os.makedirs(folder_name, exist_ok = True)
            shutil.move(image, folder_name+'/'+image.split('/')[-1])
            
        cv2.destroyAllWindows()
    except Exception as e:
        #excluded.append(image)
        print(e)


