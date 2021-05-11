

"""
Course:  Training YOLO v3 for Objects Detection with Custom Data
Section-2
Objects Detection in Real Time with YOLO v3 and OpenCV
File: yolo-3-camera.py
"""


# Detecting Objects in Real Time with OpenCV deep learning library
#
# Algorithm:
# Reading stream video from camera --> Loading YOLO v3 Network -->
# --> Reading frames in the loop --> Getting blob from the frame -->
# --> Implementing Forward Pass --> Getting Bounding Boxes -->
# --> Non-maximum Suppression --> Drawing Bounding Boxes with Labels -->
# --> Showing processed frames in OpenCV Window
#
# Result:
# Window with Detected Objects, Bounding Boxes and Labels in Real Time


# Importing needed libraries
import numpy as np
import cv2
import time
import pyrealsense2 as rs
import json


"""realsense setup"""
jsonObj = json.load(open("custom.json"))
json_string= str(jsonObj).replace("'", '\"')

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

freq=int(jsonObj['stream-fps'])
print("W: ", int(jsonObj['stream-width']))
print("H: ", int(jsonObj['stream-height']))
print("FPS: ", int(jsonObj['stream-fps']))
config.enable_stream(rs.stream.depth, int(jsonObj['stream-width']), int(jsonObj['stream-height']), rs.format.z16, int(jsonObj['stream-fps']))
config.enable_stream(rs.stream.color, int(jsonObj['stream-width']), int(jsonObj['stream-height']), rs.format.bgr8, int(jsonObj['stream-fps']))
profile = pipeline.start(config)
dev = profile.get_device()
advnc_mode = rs.rs400_advanced_mode(dev)
advnc_mode.load_json(json_string)



depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 0.38 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

"""
Start of:
Reading stream video from camera
"""

# Defining 'VideoCapture' object
# and reading stream video from camera
#camera = cv2.VideoCapture(0)

# Preparing variables for spatial dimensions of the frames
h, w = None, None

"""
End of:
Reading stream video from camera
"""


"""
Start of:
Loading YOLO v3 network
"""

# Loading COCO class labels from file
# Opening file
# Pay attention! If you're using Windows, yours path might looks like:
# r'yolo-coco-data\coco.names'
# or:
# 'yolo-coco-data\\coco.names'
path = "/home/disorn/code_save/python/python-yolo/"
with open(path+'test1/classes.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]


# # Check point
# print('List with labels names:')
# print(labels)

# Loading trained YOLO v3 Objects Detector
# with the help of 'dnn' library from OpenCV
# Pay attention! If you're using Windows, yours paths might look like:
# r'yolo-coco-data\yolov3.cfg'
# r'yolo-coco-data\yolov3.weights'
# or:
# 'yolo-coco-data\\yolov3.cfg'
# 'yolo-coco-data\\yolov3.weights'
network = cv2.dnn.readNetFromDarknet(path+'test1/test.cfg',
                                     path+'test1/test_final.weights')

network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Getting list with names of all layers from YOLO v3 network
layers_names_all = network.getLayerNames()

# # Check point
# print()
# print(layers_names_all)

# Getting only output layers' names that we need from YOLO v3 algorithm
# with function that returns indexes of layers with unconnected outputs
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# # Check point
# print()
# print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.8

# Setting threshold for filtering weak bounding boxes
# with non-maximum suppression
threshold = 0.3

# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# # Check point
# print()
# print(type(colours))  # <class 'numpy.ndarray'>
# print(colours.shape)  # (80, 3)
# print(colours[0])  # [172  10 127]

"""
End of:
Loading YOLO v3 network
"""


"""
Start of:
Reading frames in the loop
"""

# Defining loop for catching frames
while True:
    # Capturing frame-by-frame from camera
    #_, frame = camera.read()
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        continue
    colorizer = rs.colorizer(2)
    depth_image_raw = np.asanyarray(aligned_depth_frame.get_data())

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 3)
    hole_filling = rs.hole_filling_filter()
    temporal = rs.temporal_filter()
    
    depth_filter = depth_to_disparity.process(aligned_depth_frame)
    depth_filter = spatial.process(depth_filter)
    depth_filter = temporal.process(depth_filter)
    depth_filter = disparity_to_depth.process(depth_filter)
    depth_filter = hole_filling.process(depth_filter)
    depth_fill_raw=np.asanyarray(depth_filter.get_data())
    
    depth_image = np.asanyarray(colorizer.colorize(depth_filter).get_data())
    color_image = np.asanyarray(color_frame.get_data())
    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    #depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    depth_image_3d = depth_image
    depth_bg_removed_raw = np.where((depth_fill_raw > clipping_distance) | (depth_fill_raw <= 0), clipping_distance, depth_fill_raw)
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
    depth_bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, depth_image)
    depth_image_raw_8 = (depth_fill_raw/6).astype('uint8')
    # Render images
    #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    #depth_colormap = np.asanyarray(colorizer.colorize(depth_image).get_data())
    images = np.hstack((color_image , depth_image))
    
    # Getting spatial dimensions of the frame
    # we do it only once from the very beginning
    # all other frames have the same dimension
    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = color_image.shape[:2]

    """
    Start of:
    Getting blob from current frame
    """

    # Getting blob from current frame
    # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob from current
    # frame after mean subtraction, normalizing, and RB channels swapping
    # Resulted shape has number of frames, number of channels, width and height
    # E.G.:
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    blob = cv2.dnn.blobFromImage(images, 1 / 255.0, (416, 416),
                                 swapRB=False, crop=False)

    """
    End of:
    Getting blob from current frame
    """

    """
    Start of:
    Implementing Forward pass
    """

    # Implementing forward pass with our blob and only through output layers
    # Calculating at the same time, needed time for forward pass
    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Showing spent time for single current frame
    print('Current frame took {:.5f} seconds'.format(end - start))

    """
    End of:
    Implementing Forward pass
    """

    """
    Start of:
    Getting bounding boxes
    """

    # Preparing lists for detected bounding boxes,
    # obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # # Check point
            # # Every 'detected_objects' numpy array has first 4 numbers with
            # # bounding box coordinates and rest 80 with probabilities
            # # for every class
            # print(detected_objects.shape)  # (85,)

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial frame size
                # YOLO data format keeps coordinates for center of bounding box
                # and its current width and height
                # That is why we can just multiply them elementwise
                # to the width and height
                # of the original frame and in this way get coordinates for center
                # of bounding box, its width and height for original frame
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    """
    End of:
    Getting bounding boxes
    """

    """
    Start of:
    Non-maximum suppression
    """

    # Implementing non-maximum suppression of given bounding boxes
    # With this technique we exclude some of bounding boxes if their
    # corresponding confidences are low or there is another
    # bounding box for this region with higher confidence

    # It is needed to make sure that data type of the boxes is 'int'
    # and data type of the confidences is 'float'
    # https://github.com/opencv/opencv/issues/12789
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

    """
    End of:
    Non-maximum suppression
    """

    """
    Start of:
    Drawing bounding boxes and labels
    """

    # Checking if there is at least one detected object
    # after non-maximum suppression
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Getting current bounding box coordinates,
            # its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Preparing colour for current bounding box
            # and converting from numpy array to list
            colour_box_current = colours[class_numbers[i]].tolist()

            # # # Check point
            # print(type(colour_box_current))  # <class 'list'>
            # print(colour_box_current)  # [172 , 10, 127]

            # Drawing bounding box on the original current frame
            cv2.rectangle(color_image, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            # Putting text with label and confidence on the original image
            cv2.putText(color_image, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    """
    End of:
    Drawing bounding boxes and labels
    """

    """
    Start of:
    Showing processed frames in OpenCV Window
    """

    # Showing results obtained from camera in Real Time

    # Showing current frame with detected objects
    # Giving name to the window with current frame
    # And specifying that window is resizable
    cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
    # Pay attention! 'cv2.imshow' takes images in BGR format
    cv2.imshow('YOLO v3 Real Time Detections', color_image)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    """
    End of:
    Showing processed frames in OpenCV Window
    """

"""
End of:
Reading frames in the loop
"""


# Releasing camera
camera.release()
# Destroying all opened OpenCV windows
cv2.destroyAllWindows()


"""
Some comments
cv2.VideoCapture(0)
To capture video, it is needed to create VideoCapture object.
Its argument can be camera's index or name of video file.
Camera index is usually 0 for built-in one.
Try to select other cameras by passing 1, 2, 3, etc.
"""