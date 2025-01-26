import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pytesseract
import Utilities
from Utilities import load_images_from_folder, getOutputsNames, find_bounding_boxes

debug = True

def car_image_detection(network, image, width, height):
    # set neural network input
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (width, height), [0, 0, 0], 1, crop=False)
    network.setInput(blob)
    # retrieve neural network outputs and find bounding boxes
    outs = network.forward(getOutputsNames(network))
    box_image = image.copy()
    boxes, classIds, confidences = find_bounding_boxes(box_image, outs)

    if debug:
        cv2.namedWindow("bounding box car", cv2.WINDOW_NORMAL)
        cv2.imshow("bounding box car", box_image)
        cv2.waitKey(0)

    car_images = []
    for car_box in boxes:
        left,top,car_width,car_height = car_box
        car_image = image[top:top+car_height, left:left+car_width]
        car_images.append(car_image)

    return car_images

'''
Funtions that returns the set of license plates found in a set of images. The function is composed of 2 steps:
1) In the first step the function detect the set of cars in an image
2) Given the cars found in the first step, the function detects the respective license plates

'''
def detect_license_plates(images, vehicle_detection, plate_detection):
    NN_width = 416
    NN_height = 416
    for image in images:

        #Car detection
        detected_cars = car_image_detection(vehicle_detection, image, NN_height, NN_height)

        for detected_car in detected_cars:
            if debug:
                cv2.namedWindow("bounding box car", cv2.WINDOW_NORMAL)
                cv2.imshow("bounding box car", detected_car)
                cv2.waitKey(0)







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




