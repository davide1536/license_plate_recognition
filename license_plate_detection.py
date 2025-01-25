import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pytesseract
import Utilities
from Utilities import load_images_from_folder

#load input images
input_images = load_images_from_folder("Images/")

#load neural network weights - Car Detection
config_file_veic_detection = 'yolov3config_veic_detection.cfg'
model_file_veic_detection = 'yolov3_veic_detection.weights'
veic_detection_net = cv2.dnn.readNetFromDarknet(config_file_veic_detection, model_file_veic_detection)

#load neural network weights - License Plate Detection
config_file_plate_detection = 'yolov3config_plate_detection.cfg'
model_file_plate_detection = 'yolov3_plate_detection.weights'
license_plate_detection_net = cv2.dnn.readNetFromDarknet(config_file_plate_detection, model_file_plate_detection)


