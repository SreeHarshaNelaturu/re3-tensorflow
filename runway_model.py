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
#global counter


initialize = True
@runway.command("track_object", inputs = inputs, outputs = outputs, description="Track Selected Object")
def track_object(tracker, inputs):
    global initialize

    img = np.array(inputs["input_image"])

    ip_bbox = [196, 106, 408, 461] #Got it from the cv2.VideoCapture

    if initialize:
       print("Object Tracker Intialized")
       boxToDraw = ip_bbox
       print("ip_bbx",boxToDraw)
       outputBoxToDraw = tracker.track('webcam', img[:, :, ::-1], boxToDraw)
       initialize = False
    else:
        outputBoxToDraw = tracker.track('webcam', img[:, :, ::-1])
    print("op_bbox", outputBoxToDraw)
    normalize_bbox = [[outputBoxToDraw[0] / img.shape[1], outputBoxToDraw[1] / img.shape[0],
                      outputBoxToDraw[2] / img.shape[1], outputBoxToDraw[3] / img.shape[0]]]

    return {"op_bbox" : normalize_bbox}

if __name__ == "__main__":
    runway.run(model_options={"checkpoint" : "logs/checkpoints/"})






















""" Unnormalized Data Methods, can be removed once the final I/O format is decided.
        
        
        
        # print("nomralize", normalize_bbox)
        # print(type(normalize_bbox[0]))
        #return {"op_bbox" : normalize_bbox}
    """


