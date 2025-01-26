import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pytesseract
import Utilities
from Utilities import load_images_from_folder, getOutputsNames, getBoundingBoxes

debug = True

#retrieve the roi (region of interest) of a given image
def get_roi(network, image, width, height):

    box_image,boxes, classIds, confidences = getBoundingBoxes(network, image, width, height)


    roi_images = []
    for box in boxes:
        left,top,car_width,car_height = box
        roi_image = image[top:top+car_height, left:left+car_width]
        roi_images.append(roi_image)

    return roi_images

'''
Funtions that returns the set of license plates found in a set of images. The function is composed of 2 steps:
1) In the first step the function detect the set of cars in an image
2) Given the cars found in the first step, the function detects the respective license plates

'''
def detect_license_plates(images, vehicle_detection, plate_detection):
    NN_width = 416
    NN_height = 416
    total_license_plates = []
    for image in images:
        if debug:
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", image)
            cv2.waitKey(0)
        #Car detection
        detected_cars = get_roi(vehicle_detection, image, NN_width, NN_height)

        for detected_car in detected_cars:
            if debug:
                cv2.namedWindow("bounding box car", cv2.WINDOW_NORMAL)
                cv2.imshow("bounding box car", detected_car)
                cv2.waitKey(0)
            car_license_plates = get_roi(plate_detection, detected_car, NN_width, NN_height)
            if debug:
                for car_license_plate in car_license_plates:
                    cv2.namedWindow("license_plate", cv2.WINDOW_NORMAL)
                    cv2.imshow("license_plate", car_license_plate)
                    cv2.waitKey(0)
            total_license_plates.append(car_license_plates)

    return total_license_plates









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

detect_license_plates(input_images, veic_detection_net, license_plate_detection_net)




