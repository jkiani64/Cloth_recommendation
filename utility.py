import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os, time, shutil
from tqdm import tqdm
import numpy as np, pandas as pd
import cv2
from tqdm import tqdm_notebook, tqdm # Iteration visualization
from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

def convert_labels(path, x1, y1, x2, y2):
    """
    Definition: Parses label files to extract label and bounding box
        coordinates.  Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
    """
    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin
    size = get_img_shape(path)
    height, width = size[:2]
    max_height = 300
    max_width = 300
    scaling_factor = 1
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width/float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
    x1 = int(x1/scaling_factor)
    y1 = int(y1/scaling_factor)  
    x2 = int(x2/scaling_factor)  
    y2 = int(y2/scaling_factor)  
    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    dw = 1./size[1]
    dh = 1./size[0]
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
def get_img_shape(path):
    img = cv2.imread(path)
    try:
        return img.shape
    except Exception:
        raise ValueError(f'There is no {path}')
def from_yolo_to_cor(box, img_h, img_w): 
    x1, y1 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)
    x2, y2 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    return x1, y1, x2, y2
    
def draw_boxes(path, boxes, original = False):
    img = cv2.imread(path)
    print(img.shape)
    print(boxes)
    plt.figure(figsize = (12, 6))
    if original == False:
        x1, y1, x2, y2 = from_yolo_to_cor(boxes, img.shape[0], img.shape[1])
    else:
        x1, y1, x2, y2 = boxes
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 3)
    plt.subplot(1,1, 1), plt.imshow(img)
    
def cropping_images(path,x_0,y_0,width_0,heigh_0):
    bbox = [x_0,y_0,width_0,heigh_0]
    # import image
    f_image = Image.open(path)
    
    #Bounding box cordinates
    x_1, y_1, x_2, y_2 = from_yolo_to_cor(bbox, f_image.size[1], f_image.size[0])
#     plt.figure(figsize = (12, 8))    
#     img = cv2.imread(path)
#     cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (255,0,0), 3)
#     plt.subplot(1, 2,1); plt.imshow(img);
    
    box_height = y_2 - y_1
    box_width = x_2 - x_1

    # get width and height of image
    width, height = f_image.size

    # crop image randomly around bouding box within a 0.3 * bbox extra range
    rnd_n = random.random() * 0.15 + 0.05
    left = max(0, x_1 - round(rnd_n * box_width))
    
    rnd_n = (random.random() * 0.15 + 0.05)
    right = min(x_2 + round(rnd_n * box_width), width)

    
    rnd_n = random.random() * 0.15 + 0.05
    top = max(0, y_1 - round(rnd_n * box_height))
    
    rnd_n = random.random() * 0.15 + 0.05
    bottom = min(y_2 + round(rnd_n * box_height), height)
    
    # Crop the image
    f_image = f_image.crop((left, top, right, bottom))
    
    _width, _height = width, height
    width, height = f_image.size
    
    #f_image.show()
    new_path = path[:-4] + '_crop' + '.jpg'
    #f_image.save(new_path, 'jpeg')
    try:
        f_image.save(new_path, 'jpeg')
    except:
        #print(f'Used convertor for {new_path}')
        f_image = f_image.convert('RGB')
        f_image.save(new_path, 'jpeg')

    # List of normalized left x coordinates in bounding box (1 per box)
    xmins = (x_1 - left) / width
    # List of normalized right x coordinates in bounding box (1 per box)
    xmaxs = (x_2 - left) / width
    # List of normalized top y coordinates in bounding box (1 per box)
    ymins = (y_1 - top) / height
    # List of normalized bottom y coordinates in bounding box (1 per box)
    ymaxs = (y_2 - top) / height
    
    x_middle = (xmins + xmaxs) / 2
    y_middle = (ymins + ymaxs) / 2
    bbox_width = (xmaxs - xmins)
    bbox_height = (ymaxs - ymins)
    
    bbox_new = (x_middle, y_middle, bbox_width, bbox_height)

#    assert (xmins >= 0.) and (xmaxs < 1.01) and (ymins >= 0.) and (ymaxs < 1.01), print(path)
#     img = cv2.imread(new_path)   
#     x_1, y_1, x_2, y_2 = from_yolo_to_cor(bbox_new, img.shape[0], img.shape[1])
#     cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (255,0,0), 3)
#     plt.subplot(1, 2,2); plt.imshow(img);
    
    return bbox_new
def copy_all_images(directory):
    folders = os.listdir(directory)
    new_directory = os.path.join(directory, 'images')
    os.mkdir(new_directory)
    count = 0
    for fld in folders:
        try:
            folder = os.path.join(directory, fld)
            images = os.listdir(folder)
            for idx, img in enumerate(images):
                full_file_name = os.path.join(folder, img)
                img_name = fld + f'_idx{idx+1}' + '.jpg'
                dest = os.path.join(new_directory, img_name)
                shutil.copy(full_file_name, dest)
                count += 1
        except:
            print(f'There is no this directory: {fld}')
    print(f'{count} images are moved from this directory {directory}')