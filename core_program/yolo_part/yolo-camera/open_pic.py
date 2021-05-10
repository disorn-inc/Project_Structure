import numpy as np
import cv2
import time

camera = cv2.imread('/home/disorn/code_save/Project_Structure/core_program/yolo_part/yolo-camera/test1/fuse_image_bgremove_v2_1.png' , cv2.IMREAD_UNCHANGED)
frame = (camera)
cv2.imshow('j',frame)
print(camera[:,:,0:3])
cv2.waitKey(0)
cv2.destroyAllWindows() 