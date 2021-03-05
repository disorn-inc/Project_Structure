## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import json

jsonObj = json.load(open("custom.json"))
json_string= str(jsonObj).replace("'", '\"')

# Create a pipeline
pipeline = rs.pipeline()
i = 0
#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
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
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
#profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 0.7 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        colorizer = rs.colorizer(2)
        depth_image_raw = np.asanyarray(aligned_depth_frame.get_data())
        depth_image = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        #depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        depth_image_3d = depth_image
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #depth_colormap = np.asanyarray(colorizer.colorize(depth_image).get_data())
        images = np.hstack((bg_removed, depth_image))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        if key == ord("s"):
            cv2.imwrite('./color_image'+str(i)+'.png', bg_removed)
            cv2.imwrite('./depth_image'+str(i)+'.png', depth_image)
            np.save('./depth_image_raw3c'+str(i), depth_image)
            np.save('./depth_image_raw'+str(i), depth_image_raw)
            i=i+1
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
