import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import sys,os

home = os.path.expanduser("~")
path = "/home/disorn/code_save/Project_Structure/core_program/yolo_part/yolo-camera/"
colorlabel=['gray','yellow']
new_model = tf.keras.models.load_model(path + '/gg.h5')
h, w = None, None

with open(path+'/depth_hl.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet(path+'test1/rgb_combine_2color.cfg',
                                     path+'test1/rgb_combine_2color_final.weights')

layers_names_all = network.getLayerNames()
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes
# with non-maximum suppression
threshold = 0.3
ho = 0
# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

for k in range(801,921) or range(121) :
    color_image = cv2.imread("/home/kittipong/dataset_combine_2color/RGB/color_image"+str(k)+".png")
    depth_image = cv2.imread('/home/kittipong/dataset_combine_2color/depth_image/depth_image'+str(k)+".png")
    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = depth_image.shape[:2]
    blob = cv2.dnn.blobFromImage(depth_image, 1/255 , (608, 608),
                                 swapRB=False, crop=False)  
    network.setInput(blob)  # setting blob as input to the network
    output_from_network = network.forward(layers_names_output)
    bounding_boxes = []
    confidences = []
    class_numbers = []
    for result in output_from_network:
        #print(np.round(result,2))
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            #print(np.round(scores,2))
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]
            #print(confidence_current)
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
                #print(box_height)
                if(x_min<0):
                    x_min1=0
                    box_width1=box_width+x_min
                else:
                    x_min1=x_min
                    box_width1=box_width
                if(y_min<0):
                    y_min1=0
                    box_height1=box_height+y_min
                else:
                    y_min1=y_min
                    box_height1=box_height
                color_segment = color_image[int(y_min1):int(y_min1+box_height1),int(x_min1):int(x_min1+box_width1)]
                #print(color_segment)
                color_segment = cv2.resize(color_segment,(180,180))
                color_segment = cv2.cvtColor(color_segment, cv2.COLOR_BGR2RGB)
                color_segment = np.expand_dims(color_segment, axis=0)
                #print(color_segment.shape)
                predictions=new_model.predict(color_segment)
                #print(predictions)
                color_class=np.argmax(predictions)
                #print(colorlabel[color_class])
                # Adding results into prepared lists
                #print(depth_fill_raw[int(y_center),int(x_center)])
                #vis_depth.append(float(depth_image_raw_8[int(y_center),int(x_center)]))
                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height),x_center, y_center])
                confidences.append(float(confidence_current))
                class_numbers.append([class_current,color_class])
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)    
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Getting current bounding box coordinates,
            # its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            x_center, y_center = bounding_boxes[i][4], bounding_boxes[i][5]
            # Preparing colour for current bounding box
            # and converting from numpy array to list
            colour_box_current = colours[class_numbers[i][0]].tolist()

            # # # Check point
            # print(type(colour_box_current))  # <class 'list'>
            # print(colour_box_current)  # [172 , 10, 127]

            # Drawing bounding box on the original current frame
            cv2.circle(color_image,(int(x_center),int(y_center)), 2, (255,255,255), -1)
            cv2.rectangle(color_image, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i][0])]+colorlabel[int(class_numbers[i][1])]+'dome',
                                                   confidences[i])

            # Putting text with label and confidence on the original image
            cv2.putText(color_image, text_box_current, (x_min, y_min+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
    print(k)
    cv2.imwrite("/home/kittipong/dataset_combine_2color/test_image/result_image"+str(k)+".png",color_image)