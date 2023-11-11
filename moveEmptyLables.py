import os
import shutil

files = '/home/shussain/Downloads/FireSmokeDetection/out'
print(len(files))
dest = '/home/shussain/Downloads/FireSmokeDetection/move'

for i in os.listdir(files):
    if i.endswith('.txt'):
        label_path = os.path.join(files,i)
        img_path = os.path.join(files,i.replace('.txt', '.jpg'))
        
        if os.path.getsize(label_path) == 0:
            print(len(label_path))
            shutil.move(label_path, os.path.join(dest, i))
            shutil.move(img_path, os.path.join(dest, i.replace('.txt', '.jpg')))
            
        print('moving all files')
