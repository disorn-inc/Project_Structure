import glob
import xml.etree.ElementTree as ET

path_xml = '/home/disorn/code_save/Yolo2Pascal-annotation-conversion/test/'

def write_txt(gt,file_name_text):
    path_gt = 'groundtruths/'
    with open(path_gt + file_name_text, 'w') as f:
        for row in gt:
            f.writelines(' '.join(map(str, row)) + '\n')
        print(1)

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
    print('outloop',gt)
    print(file_name_text)
    write_txt(gt,file_name_text)
    
    #label = 
    

for xmlfile in glob.glob(path_xml + "*.xml"):
    extract_from_xml(xmlfile)