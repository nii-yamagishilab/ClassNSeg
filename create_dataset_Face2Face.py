"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for making FaceForensics Face2Face dataset
"""

import argparse
import cv2
import numpy as np
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--dataset', default ='datasets/FaceForensics/source-to-target', help='path to dataset')
# parser.add_argument('--dataset', default ='datasets/FaceForensics/selfreenactment', help='path to dataset')
parser.add_argument('--original', default ='original', help='original videos')
parser.add_argument('--mask', default ='mask', help='mask videos')
parser.add_argument('--altered', default ='altered', help='altered video')
parser.add_argument('--num_frames', type=int, default=200, help='Number of frames extracted for each video')
parser.add_argument('--output', default = 'datasets/full', help= 'name of output folder')
parser.add_argument('--scale', type=float, default =1.3, help='enables resizing')

opt = parser.parse_args()
print(opt)

def to_bw(mask, thresh_binary=127, thresh_otsu=255):
    im_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, thresh_binary, thresh_otsu, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #invert
    return cv2.bitwise_not(im_bw)

def get_bbox(mask, thresh_binary=127, thresh_otsu=255):
    im_bw = to_bw(mask, thresh_binary, thresh_otsu)

    im2, contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

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

def extract_face_videos(input_path, output_path):
    f_vid_original = os.path.join(input_path, opt.original)
    f_vid_mask = os.path.join(input_path, opt.mask)
    f_vid_altered = os.path.join(input_path, opt.altered)

    f_img_original = os.path.join(output_path, opt.original)
    f_img_altered = os.path.join(output_path, opt.altered)

    blank_img = np.zeros((opt.imageSize,opt.imageSize,3), np.uint8)

    for f in os.listdir(f_vid_mask):
        if os.path.isfile(os.path.join(f_vid_mask, f)):
            if f.lower().endswith(('avi')):
                print(f)
                filename = os.path.splitext(f)[0]

                vidcap_original = cv2.VideoCapture(os.path.join(f_vid_original, f))
                success_original, image_original = vidcap_original.read()

                vidcap_mask = cv2.VideoCapture(os.path.join(f_vid_mask, f))
                success_mask, image_mask = vidcap_mask.read()

                vidcap_altered = cv2.VideoCapture(os.path.join(f_vid_altered, f))
                success_altered, image_altered = vidcap_altered.read()

                count = 0

                while (success_original and success_mask and success_altered):
                    bbox = get_bbox(image_mask)

                    if bbox is None:
                        count += 1
                        continue

                    original_cropped = extract_face(image_original, bbox, opt.scale)
                    altered_cropped = extract_face(image_altered, bbox, opt.scale)

                    mask_cropped = to_bw(extract_face(image_mask, bbox, opt.scale))
                    mask_cropped = np.stack((mask_cropped,mask_cropped, mask_cropped), axis=2)

                    if (original_cropped is not None) and (altered_cropped is not None) and (mask_cropped is not None):
                        original_cropped = np.concatenate((original_cropped, blank_img), axis=1)
                        altered_cropped = np.concatenate((altered_cropped, mask_cropped), axis=1)

                        cv2.imwrite(os.path.join(f_img_original, filename + "_%d.jpg" % count), original_cropped)
                        cv2.imwrite(os.path.join(f_img_altered, filename + "_%d.jpg" % count), altered_cropped)

                        count += 1

                    if count >= opt.num_frames:
                        break

                    success_original, image_original = vidcap_original.read()
                    success_mask, image_mask = vidcap_mask.read()
                    success_altered, image_altered = vidcap_altered.read()

def extract_face_datasets(dataset, output, action = ('train', 'test', 'validation')):
    input_path = os.path.join(dataset, 'c23', 'test')
    output_path = os.path.join(output, 'c23', 'test')

    extract_face_videos(input_path, output_path)

if __name__ == '__main__':
    extract_face_datasets(opt.dataset, opt.output)
