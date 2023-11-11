from PIL import Image, ImageDraw, ImageFilter
import cv2
'''
im1 = Image.open('/home/shussain/Downloads/1.png')
print('image1', im1)
print(im1.shape)
im2 = Image.open('/home/shussain/Downloads/94.png')

#im1.paste(im2)
#im1.save('pillow_paste.png', quality=95)

#back_im = im1.copy()
#back_im.paste(im2, (100, 50))
#back_im.save('paste_pos.jpg', quality=95)

mask_im = Image.new("L", im2.size, 0)
draw = ImageDraw.Draw(mask_im)
draw.rectangle((140, 50, 260, 170), fill=255)
mask_im.save('mask_circle.jpg', quality=95)

back_im = im1.copy()
back_im.paste(im2, (0, 0), mask_im)
back_im.save('paste_mask_circle.jpg', quality=95)

'''
import cv2

im = cv2.imread('/home/shussain/Downloads/1.png')

print(type(im))
# <class 'numpy.ndarray'>

print(im.shape)
print(type(im.shape))
