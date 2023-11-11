from rembg import remove
from PIL import Image

imput_img = '/home/shussain/Downloads/img.jpg'
output = 'img.png'

img = Image.open(imput_img)
out = remove(img)

out.save(output)

