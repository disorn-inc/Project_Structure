import glob
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import time


'''path to xml groundtruths file, this generate from yolo to voc[xml fomat] '''
path_xml_depth = '/home/disorn/code_save/Yolo2Pascal-annotation-conversion/test_depth/'
path_xml_rgb = '/home/disorn/code_save/Yolo2Pascal-annotation-conversion/test_rgb/'
path = "/home/disorn/code_save/Project_Structure/core_program/yolo_part/yolo-camera/"
new_model = tf.keras.models.load_model(path + '/gg.h5')
"""function to write .txt file"""
def write_txt(gt,file_name_text):
    path_gt = '/home/disorn/metrics_measurement/test_rgb/detections/'
    with open(path_gt + file_name_text, 'w') as f:
        for row in gt:
            f.writelines(' '.join(map(str, row)) + '\n')
            
def write_img(img,filename):
    path_save_result = '/home/disorn/metrics_measurement/test_rgb/result_image/'
    print(path_save_result + 'result'+filename)
    cv2.imwrite(path_save_result + 'result'+filename, img)

"""function to ETL process xml file to list[] of groundtruths[label, xmin, ymin, xmax, ymax]"""
def extract_from_xml(file_to_process):
    gt = []
    tree = ET.parse(file_to_process)
    root = tree.getroot()
    file_name = root.find('filename').text
    file_name_text = file_name.replace("png","txt")
    image_BGR = cv2.imread(path_xml_depth + file_name , cv2.IMREAD_UNCHANGED) 
    h, w = image_BGR.shape[:2]
    blob = cv2.dnn.blobFromImage(image_BGR, 1/255.0 , (608,608),
                             swapRB=False, crop=False)
    with open(path+'test1/depth_assem2color.names') as f:
        labels = [line.strip() for line in f]
        
    network = cv2.dnn.readNetFromDarknet(path+'test1/rgb_combine_2color.cfg',
                                        path+'test1/rgb_combine_2color_final.weights')
    layers_names_all = network.getLayerNames()
    layers_names_output = \
        [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
    probability_minimum = 0.5
    threshold = 0.3
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()
    bounding_boxes = []
    confidences = []
    class_numbers = []
    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)
    counter = 1
    detect = []
    detect_yolo = []
    if len(results) > 0:
        for i in results.flatten():
            print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))
            counter += 1
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box_current = colours[class_numbers[i]].tolist()
            cv2.rectangle(image_BGR, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])
            cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, colour_box_current, 2)
            detect.append([labels[int(class_numbers[i])], confidences[i], x_min, y_min, x_min + box_width, y_min +box_height])
    write_txt(detect,file_name_text)
    write_img(image_BGR[:,:,0:3], file_name)
    

        #print(gt)
    #print('outloop',gt)
    #print(file_name_text)
    #write_txt(gt,file_name_text)
     
    
"""for loop for load all xml file"""
for xmlfile in glob.glob(path_xml_depth + "*.xml"):
    extract_from_xml(xmlfile)