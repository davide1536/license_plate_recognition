import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pytesseract
import Utilities
from Utilities import load_images_from_folder, getBoundingBoxes, clear_image, transform_image

def extract_text_from_license_plate(license_plates):
    for license_plate in license_plates:
        grey_plate = cv2.cvtColor(license_plate[0], cv2.COLOR_BGR2GRAY)
        binary_plate_warped = cv2.threshold(grey_plate, 0, 255,
                                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_plate_warped_cleaned = cv2.morphologyEx(binary_plate_warped, cv2.MORPH_CLOSE, squareKern)
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_plate_warped_cleaned = cv2.morphologyEx(binary_plate_warped_cleaned, cv2.MORPH_OPEN, squareKern)

        #define tesseract parameters
        psm = 7
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(psm)
        #return text
        text = pytesseract.image_to_string(binary_plate_warped_cleaned, config=options)
        license_plate.append(text)
    return license_plates

def clean_and_transform_plates(license_plates):
    debug = False
    for license_plate in license_plates:
        #retrieve license plate image
        license_plate_image = license_plate[0]
        license_plate_image =  clear_image(license_plate_image, 5, 100, 100, 240)
        if debug:
            cv2.namedWindow("cleared license plate", cv2.WINDOW_NORMAL)
            cv2.imshow("cleared license plate", license_plate_image)
            cv2.waitKey(0)
        license_plate_image = transform_image(license_plate_image)
        if debug:
            cv2.namedWindow("final license plate", cv2.WINDOW_NORMAL)
            cv2.imshow("final license plate", license_plate_image)
            cv2.waitKey(0)
        license_plate[0] = license_plate_image

    return license_plates





#retrieve the roi (region of interest) of a given image
def get_roi(network, image, width, height, offsetx, offsety):

    box_image,boxes, classIds, confidences = getBoundingBoxes(network, image, width, height)

    roi_images = []
    for box in boxes:
        left,top,car_width,car_height = box
        roi_image = image[top:top+car_height+offsety, left:left+car_width+offsetx]
        roi_images.append(roi_image)

    return roi_images, classIds, confidences

'''
Funtions that returns the set of license plates found in a set of images. The function is composed of 2 steps:
1) In the first step the function detect the set of cars in an image
2) Given the cars found in the first step, the function detects the respective license plates

'''
def detect_license_plates(images, vehicle_detection, plate_detection, license_plate_resolution_threshold=80):
    debug = False
    NN_width = 416
    NN_height = 416
    total_license_plates = []
    for image in images:
        if debug:
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", image)
            cv2.waitKey(0)
        #Car detection
        cars_info = get_roi(vehicle_detection, image, NN_width, NN_height, 0, 0)
        print(cars_info[2])
        for i,detected_car in enumerate(cars_info[0]):
            vehicle_id = cars_info[1]
            if debug:
                cv2.namedWindow("bounding box car", cv2.WINDOW_NORMAL)
                cv2.imshow("bounding box car", detected_car)
                cv2.waitKey(0)
            license_plates_info = get_roi(plate_detection, detected_car, NN_width, NN_height, 3, 3)
            if debug:
                for car_license_plate in license_plates_info[0]:
                    cv2.namedWindow("license_plate", cv2.WINDOW_NORMAL)
                    cv2.imshow("license_plate", car_license_plate)
                    cv2.waitKey(0)
            license_plates_images = license_plates_info[0]
            for license_plate_image in license_plates_images:
                if license_plate_image.shape[1] >= license_plate_resolution_threshold:
                    total_license_plates.append([license_plate_image, vehicle_id])

    return total_license_plates








if __name__ == "__main__":
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

    #retrieves the set of license plates in a given image
    license_plates_raw = detect_license_plates(input_images, veic_detection_net, license_plate_detection_net)
    license_plates_cleaned = clean_and_transform_plates(license_plates_raw)
    license_plate_with_text = extract_text_from_license_plate(license_plates_cleaned)

    for licese_plate in license_plate_with_text:
        print(licese_plate[2])
















