import glob
import xml.etree.ElementTree as ET

'''path to xml groundtruths file, this generate from yolo to voc[xml fomat] '''
path_xml = '/home/disorn/code_save/Yolo2Pascal-annotation-conversion/real_final_depth/'


"""function to write .txt file"""
def write_txt(gt,file_name_text):
    path_gt = '/home/disorn/metrics_measurement/real_final_depth/groundtruths/'
    with open(path_gt + file_name_text, 'w') as f:
        for row in gt:
            f.writelines(' '.join(map(str, row)) + '\n')

"""function to ETL process xml file to list[] of groundtruths[label, xmin, ymin, xmax, ymax]"""
def extract_from_xml(file_to_process):
    gt = []
    tree = ET.parse(file_to_process)
    root = tree.getroot()
    file_name = root.find('filename').text
    file_name_text = file_name.replace("png","txt")
    for item in root.findall('object'):
        label = item.find('name').text
        box = item.findall('bndbox')
        for coor in box:
            xmin = coor.find('xmin').text
            ymin = coor.find('ymin').text
            xmax = coor.find('xmax').text
            ymax = coor.find('ymax').text
            gt.append([label, xmin, ymin, xmax, ymax])
        
        #print(gt)
    #print('outloop',gt)
    #print(file_name_text)
    write_txt(gt,file_name_text)
     
    
"""for loop for load all xml file"""
for xmlfile in glob.glob(path_xml + "*.xml"):
    extract_from_xml(xmlfile)