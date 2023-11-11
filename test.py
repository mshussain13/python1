"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import math

from craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)
result_folder = './result/'
crop_save_folder = '/home/hsumant/Documents/CRAFTtest/Save/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    # if cuda:
    #     x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def crop_parent_img(image, image_path):

    folder_n = args.test_folder + '/'
    json_n = image_path.split('/')[-1].split(".")[0]
    with open(folder_n + json_n + '.json') as file:
        data = json.load(file)
    coords = [data[0]['vtLpMinX'], data[0]['vtLpMinY'], data[0]['vtLpMaxX'], data[0]['vtLpMaxY']]
    # print(coords)

    xmin, ymin = int(coords[0]), int(coords[1])
    xmax, ymax = int(coords[2]), int(coords[3])
    image = image[ymin:ymax, xmin:xmax]
    
    return image


def angle_between_x_axis_and_line(point1, point2):
    """Calculates the angle between the x-axis and a line defined by two points."""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.degrees(math.atan2(dy, dx))

 
def order_points_old(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def angles_for_quadrilaterals(quadrilaterals):
    """Calculates the angles between the x-axis and the lines passing through
    (x1,y1)/(x2,y2) and (x3,y3)/(x4,y4) in a list of quadrilaterals."""
    angles1 = []
    angles2 = []
    for quad in quadrilaterals:
        quad = order_points_old(quad)
        x1, y1, x2, y2, x3, y3, x4, y4 = quad[0][0],quad[0][1],quad[1][0],quad[1][1],quad[2][0],quad[2][1],quad[3][0],quad[3][1]
        angle1 = angle_between_x_axis_and_line((x1, y1), (x2, y2))
        angle2 = angle_between_x_axis_and_line((x3, y3), (x4, y4))
        angles1.append(angle1)
        angles2.append(angle2)
      
    return angles1,angles2
    # print(polys)


def rotate_image(angle_list, img_path):
        print(angle_list)
        try:
            angle = angle_list[0][0]
            
            # print(angle)
            folder_n = args.test_folder + '/'
            json_n = img_path.split('/')[-1].split(".")[0]
            with open(folder_n + json_n + '.json') as file:
                data = json.load(file)
            coords = [data[0]['vtLpMinX'], data[0]['vtLpMinY'], data[0]['vtLpMaxX'], data[0]['vtLpMaxY']]
            # print(coords)

            xmin, ymin = coords[0], coords[1]
            xmax, ymax = coords[2], coords[3]
            img_p = image_path.split('/')[-1]

            img = cv2.imread(folder_n + img_p)
            # image = cv2.imread("/home/hsumant/Desktop/programs/aNgle_rOtate/cropped/cropped_1.jpg")
            # cv2.imshow("Original", image)
            (h, w) = img.shape[:2]
            (cX, cY) = (w // 2, h // 2)

            # rotate our image by x degrees around the center of the image
            M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            bounding_box = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])  
            bounding_box = bounding_box.reshape((-1, 1, 2)) # Reshape the bounding box to a column vector
            rotated_bounding_box = cv2.transform(bounding_box, M) # Apply rotation to the bounding box coordinates
             
            # Find the new bounding box coordinates
            new_xmin, new_ymin = np.min(rotated_bounding_box[:, 0, 0]), np.min(rotated_bounding_box[:, 0, 1])
            new_xmax, new_ymax = np.max(rotated_bounding_box[:, 0, 0]), np.max(rotated_bounding_box[:, 0, 1])

            cropped_image = rotated[int(new_ymin) - 20:int(new_ymax) + 20, 
                                    int(new_xmin) - 10:int(new_xmax) + 10]
            # cv2.imshow(f"Rotated by {angle} Degrees", cropped_image)
            cv2.imwrite(crop_save_folder + f'{json_n}_r.jpg', cropped_image)
            print("Image Saved Sucessfully!")
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        except:
            print("None in predictions")

        # return coords


if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    # if args.cuda:
    #     net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    # else:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    # if args.cuda:
    #     net = net.cuda()
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        # if args.cuda:
        #     refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
        #     refine_net = refine_net.cuda()
        #     refine_net = torch.nn.DataParallel(refine_net)
        # else:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # print(image_list)
    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        image = crop_parent_img(image, image_path) # Crop the parent(big) 
                                                   # image to just the AOI i.e numberplate area

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, 
                                             args.link_threshold, args.low_text, 
                                             args.poly, refine_net) # Pass the image to 
                                                                    # CRAFT for text detection
        
        angle_info = angles_for_quadrilaterals(polys) # Calculate angle
        rotate_image(angle_info, image_path) # Rotate and Crop the parent image
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)
    

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

        continue

    print("elapsed time : {}s".format(time.time() - t))


# Threshold values(done), folder_n to args.test_folder(done), 
# seprate function for cropping image that is sent to CRAFT(done)