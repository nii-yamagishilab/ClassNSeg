"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for making FaceForensics++ DeepFakes dataset
"""

import argparse
import cv2
import numpy as np
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input_real', default ='datasets/FaceForensicsPP/test/c23/original')
parser.add_argument('--input_fake', default ='datasets/FaceForensicsPP/test/c23/deepfakes')
parser.add_argument('--mask', default ='datasets/FaceForensicsPP/test/masks/manipulated_sequences/Deepfakes/raw/masks')
parser.add_argument('--output_real', default ='datasets/deepfakes/test/original')
parser.add_argument('--output_fake', default ='datasets/deepfakes/test/altered')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--limit', type=int, default=10, help='number of images to extract for each video')
parser.add_argument('--scale', type=float, default =1.3, help='enables resizing')

opt = parser.parse_args()
print(opt)

def to_bw(mask, thresh_binary=10, thresh_otsu=255):
    im_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, thresh_binary, thresh_otsu, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return im_bw

def get_bbox(mask, thresh_binary=127, thresh_otsu=255):
    im_bw = to_bw(mask, thresh_binary, thresh_otsu)

    # im2, contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    locations = np.array([], dtype=np.int).reshape(0, 5)

    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
        else:
            cX = 0
        if M["m00"] > 0:
            cY = int(M["m01"] / M["m00"])
        else:
            cY = 0

        # calculate the rectangle bounding box
        x,y,w,h = cv2.boundingRect(c)
        locations = np.concatenate((locations, np.array([[cX, cY, w, h, w + h]])), axis=0)

    max_idex = locations[:,4].argmax()
    bbox = locations[max_idex, 0:4].reshape(4)
    return bbox

def extract_face(image, bbox, scale = 1.0):
    h, w, d = image.shape
    radius = int(bbox[3] * scale / 2)

    y_1 = bbox[1] - radius
    y_2 = bbox[1] + radius
    x_1 = bbox[0] - radius
    x_2 = bbox[0] + radius

    if x_1 < 0:
        x_1 = 0
    if y_1 < 0:
        y_1 = 0
    if x_2 > w:
        x_2 = w
    if y_2 > h:
        y_2 = h

    crop_img = image[y_1:y_2, x_1:x_2]

    if crop_img is not None:
        crop_img = cv2.resize(crop_img, (opt.imageSize, opt.imageSize))

    return crop_img

def extract_face_videos(input_real, input_fake, input_mask, output_real, output_fake):

    blank_img = np.zeros((opt.imageSize,opt.imageSize,3), np.uint8)

    for f in os.listdir(input_fake):
        if os.path.isfile(os.path.join(input_fake, f)):
            if f.lower().endswith(('mp4')):
                print(f)
                filename = os.path.splitext(f)[0]

                vidcap_real = cv2.VideoCapture(os.path.join(input_real, filename[0:3] + '.mp4'))
                success_real, image_real = vidcap_real.read()

                vidcap_fake = cv2.VideoCapture(os.path.join(input_fake, f))
                success_fake, image_fake = vidcap_fake.read()

                image_mask = cv2.imread(os.path.join(input_mask, filename, '0000.png'))

                count = 0

                while (success_real and success_fake):

                    bbox = get_bbox(image_mask)

                    if bbox is None:
                        count += 1
                        continue

                    original_cropped = extract_face(image_real, bbox, opt.scale)
                    altered_cropped = extract_face(image_fake, bbox, opt.scale)

                    mask_cropped = to_bw(extract_face(image_mask, bbox, opt.scale))
                    mask_cropped = np.stack((mask_cropped,mask_cropped, mask_cropped), axis=2)

                    if (original_cropped is not None) and (altered_cropped is not None) and (mask_cropped is not None):
                        original_cropped = np.concatenate((original_cropped, blank_img), axis=1)
                        altered_cropped = np.concatenate((altered_cropped, mask_cropped), axis=1)

                        cv2.imwrite(os.path.join(output_real, filename + "_%d.jpg" % count), original_cropped)
                        cv2.imwrite(os.path.join(output_fake, filename + "_%d.jpg" % count), altered_cropped)

                        count += 1

                    if count >= opt.limit:
                        break

                    success_real, image_real = vidcap_real.read()
                    success_fake, image_fake = vidcap_fake.read()
                    image_mask = cv2.imread(os.path.join(input_mask, filename, str(count).zfill(4) + '.png'))

if __name__ == '__main__':
    extract_face_videos(opt.input_real, opt.input_fake, opt.mask, opt.output_real, opt.output_fake)
