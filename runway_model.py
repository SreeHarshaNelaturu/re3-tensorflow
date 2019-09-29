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



@runway.command("track_object", inputs = inputs, outputs = outputs, description="Track Selected Object")
def track_object(tracker, inputs):
    global initialize
    initialize = True

    img = np.array(inputs["input_image"])
    init_bbox = [0.491131,0.343612, 0.667406,0.505140 ]

    print("This should run only once")
    if initialize:
       #Initally for the tracker to start, requires input bounding box on Frame #1
       print("Object Tracker Intialized")
       boxToDraw = init_bbox
       outputBoxToDraw = tracker.track('webcam', img[:, :, ::-1], boxToDraw)
       initialize = False #This does not work and needs to be looked into
    else:
        """The else condition is never going to actually be executed, 
           as it isn't met at any time but the objective is to be 
           able to run the tracker in prediction mode after frame 0"""
        outputBoxToDraw = tracker.track('webcam', img[:, :, ::-1])

    outputBoxToDraw = [outputBoxToDraw]
    print(outputBoxToDraw)
    return {"op_bbox" : outputBoxToDraw}

if __name__ == "__main__":
    runway.run(model_options={"checkpoint" : "logs/checkpoints/"})






















""" Unnormalized Data Methods, can be removed once the final I/O format is decided.
        #init_bbox = [160, 143, 460, 375]
        #boxToDraw = [init_bbox[0]/img.shape[0], init_bbox[1]/img.shape[1], init_bbox[2]/img.shape[0], init_bbox[3]/img.shape[1]]
        # normalize_bbox = [[outputBoxToDraw[0]/img.shape[0], outputBoxToDraw[1]/img.shape[1], outputBoxToDraw[2]/img.shape[0], outputBoxToDraw[3]/img.shape[1]]]
        # print("nomralize", normalize_bbox)
        # print(type(normalize_bbox[0]))
        #return {"op_bbox" : normalize_bbox}
    """


