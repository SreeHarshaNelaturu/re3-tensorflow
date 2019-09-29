import cv2
import argparse
import glob
import numpy as np
import os
import time
import sys
import runway
from runway.data_types import *
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker


np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


@runway.setup(options={"checkpoint" : file(is_directory=True)})
def setup(opts):
    tracker = re3_tracker.Re3Tracker(opts["checkpoint"])
    return tracker

inputs = {"input_image" : image}
outputs = {"op_bbox" : array(image_bounding_box)}

@runway.command("track_object", inputs = inputs, outputs = outputs, description="Track Selected Object")
def track_object(tracker, inputs):
    img = np.array(inputs["input_image"])
    init_bbox = [160, 143, 460, 375]
    #init_bbox = [0.491131,0.343612, 0.667406,0.505140 ]
    global initialize, outputBoxToDraw
    #initialize = True
    initialize = True
    if initialize:
        boxToDraw = [init_bbox[0]/img.shape[0], init_bbox[1]/img.shape[1], init_bbox[2]/img.shape[0], init_bbox[3]/img.shape[1]]
        print("This should run only once")
        #boxToDraw = init_bbox
        outputBoxToDraw = tracker.track('webcam', img[:, :, ::-1], boxToDraw)
        initialize = False

    print("Tracking")
    outputBoxToDraw = tracker.track('webcam', img[:,:,::-1])

    outputBoxToDraw = [outputBoxToDraw]

    return {"op_bbox" : outputBoxToDraw}

    # normalize_bbox = [[outputBoxToDraw[0]/img.shape[0], outputBoxToDraw[1]/img.shape[1], outputBoxToDraw[2]/img.shape[0], outputBoxToDraw[3]/img.shape[1]]]
    # print("nomralize", normalize_bbox)
    # print(type(normalize_bbox[0]))
    #return {"op_bbox" : normalize_bbox}
if __name__ == "__main__":
    runway.run(model_options={"checkpoint" : "logs/checkpoints/"})






