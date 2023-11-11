import os, cv2
import os.path
import numpy as np
import shutil
from glob import glob

folder_path = "./"

images = glob('/home/shussain/Downloads/Audi/*.jpg')
print(len(images))
excluded = []
for image in images:
    print(image)
    try:
        show = cv2.imread(image)
        img = cv2.resize(show, (1080, 600))
        cv2.imshow(image, img)
        k = cv2.waitKey(0)
        if k == ord('a'):
            print('army')
            folder_name = 'army'
            os.makedirs(folder_name, exist_ok = True)
            shutil.copy(image, folder_name+'/'+image.split('/')[-1])

        elif k == ord('c'):
            print('civilian')
            folder_name = 'civilian'
            os.makedirs(folder_name, exist_ok = True)
            shutil.copy(image, folder_name+'/'+image.split('/')[-1])

        cv2.destroyAllWindows()
    except Exception as e:
        #excluded.append(image)
        print(e)
