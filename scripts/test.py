# Python import 
import os
import time
import numpy as np
import cv2
import math
from shutil import rmtree
# self import
from main.model.squeezeDet import  SqueezeDet
from main.config.create_config import load_dict
from main.model.evaluation import filter_batch

test_path = "/home/spiderkiller/squeezedet-keras/training/tmp/"
# test_path = "/home/spiderkiller/squeezedet-keras/training/tmp/"
# Output
output_path = "/home/spiderkiller/squeezedet-keras/training/result/"
debug_path = "/mnt/c/Users/spide/Desktop/tmp/"
DEBUG = True

def pred2bbox(y_pred, config):
    """
    Arguments:
        y_pred {[type]} -- tensor of predictions
        config {[type]} -- dict of various hyperparameters
    Returns:
        [type] -- dict of various hyperparameters
    """
    #filter batch with nms
    all_filtered_boxes, all_filtered_classes, all_filtered_scores = filter_batch(y_pred, config)

    # for i, img_name in enumerate(images_name):
    ans = []
    i = 0
    for j, det_box in enumerate(all_filtered_boxes[i]):
        #transform into xmin, ymin, xmax, ymax
        det_box = bbox_transform_single_box(det_box)
        #add rectangle and text
        ans.append([det_box[0],
                    det_box[1],
                    det_box[2],
                    det_box[3],
                    config.CLASS_NAMES[all_filtered_classes[i][j]],
                    all_filtered_scores[i][j]])
    return ans

def bbox_transform_single_box(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = int(np.floor(cx-w/2))
    out_box[1] = int(np.floor(cy-h/2))
    out_box[2] = int(np.floor(cx+w/2))
    out_box[3] = int(np.floor(cy+h/2))

    return out_box

images_paths = [os.path.join(test_path, i) for i in os.listdir(test_path)]
print("Total images = " + str(len(images_paths)))

# Enable GPU?!
os.environ['CUDA_VISIBLE_DEVICES'] = ""
#create config object
cfg = load_dict("squeeze_test.config")
# Init model
squeeze = SqueezeDet(cfg)
# Load weight
# squeeze.model.load_weights("/home/spiderkiller/squeezedet-keras/main/model/kitti.hdf5")
# squeeze.model.load_weights("/home/spiderkiller/squeezedet-keras/experiments/kitti/log/checkpoints/model.100-1.18.hdf5")
squeeze.model.load_weights("/home/spiderkiller/squeezedet-keras/experiments/kitti/log/checkpoints/model.2230-8.57.hdf5")
# squeeze.model.load_weights("/home/spiderkiller/squeezedet-keras/main/model/imagenet.hdf5")

# Clean output path
rmtree(debug_path, ignore_errors=True)
os.mkdir(debug_path)
rmtree(output_path, ignore_errors=True)
os.mkdir(output_path)

for img_path in images_paths:
    img = cv2.resize(cv2.imread(img_path), (1248, 384))
    img = np.array([img])
    
    # Predict
    t_start = time.time()
    y_pred = squeeze.model.predict(img)# , batch_size=1)
    print("Time : " + str(time.time() - t_start) + " sec.")
    
    # get bbox
    dets = pred2bbox(y_pred, cfg)

    # for img_path in imgs_name:
    n = os.path.split(img_path)[1].split('.')[0]
    if DEBUG:
        img = cv2.imread(img_path)
    with open(os.path.join(output_path, n + '.txt'), 'w') as f:
        string = ''
        for det in dets:
            string += det[4] + " -1 -1 -1 " + str(det[0]) + " " + str(det[1]) + " " + str(det[2]) + " " + str(det[3]) + " -1 -1 -1 -1 -1 -1 -1 " + str(det[5]) + '\n'
            if DEBUG:
                cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(img, det[4] + " " + str(det[5]), (det[0], det[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 1, cv2.LINE_AA)
        f.write(string)
    
    if DEBUG:
        cv2.imwrite("/mnt/c/Users/spide/Desktop/tmp/" + n + ".png", img)
