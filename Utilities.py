import cv2
import os
import numpy as np
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def findObjectsInFrame(frame, outs, objectnessThreshold=0.5, confThreshold = 0.5, nmsThreshold = 0.3, debug = False):
    """Remove the bounding boxes with low confidence using non-maxima suppression."""
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []

    boxes_filtered = []
    classIds_filtered = []
    confidences_filtered = []

    # Loop through all outputs.
    for out in outs:
        for detection in out:
            if detection[4] > objectnessThreshold:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)

                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        box = boxes[i]
        boxes_filtered.append(box)
        classIds_filtered.append(classIds[i])
        confidences_filtered.append(confidences[i])
        if debug:
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 255, 255), 2)
        # label = "{}:{:.2f}".format(classes[classIds[i]], confidences[i])
        # display_text(frame, label, left, top)
    return boxes_filtered, classIds_filtered, confidences_filtered


def getOutputsNames(net):
    """Get the names of all output layers in the network."""
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def getBoundingBoxes(network, image, width, height):
    # set neural network input
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (width, height), [0, 0, 0], 1, crop=False)
    network.setInput(blob)
    # retrieve neural network outputs and find bounding boxes
    outs = network.forward(getOutputsNames(network))
    box_image = image.copy()
    boxes, classIds, confidences = findObjectsInFrame(box_image, outs)
    return box_image, boxes, classIds, confidences

def clear_image(image, dia, sigmaColor, sigmaSpace, target_x):
    image = cv2.bilateralFilter(image, dia, sigmaColor, sigmaSpace)

    if (image.shape[1] < target_x):
        scale_factor = round(target_x / image.shape[1])
        # perform upscaling
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    return image


def fourPointsTransform(frame, vertices):
    """Extracts and transforms roi of frame defined by vertices into a rectangle."""
    # Get vertices of each bounding box
    vertices = np.asarray(vertices).astype(np.float32)
    outputSize = (frame.shape[1], frame.shape[0])
    targetVertices = np.array([
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1],
        [0, outputSize[1] - 1]
        ], dtype="float32")
    # Apply perspective transform
    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result


def sortPoint(matrix):
    ordered_matrix = matrix[np.argsort(matrix[:, 1])[::-1]]  # Ordina per y in ordine decrescente

    bottom = ordered_matrix[:2]  # I primi due punti sono quelli con y più basso
    top = ordered_matrix[2:]  # Gli altri due punti sono quelli con y più alto


    odered_top = top[np.argsort(top[:, 0])]

    ordered_bottom = bottom[np.argsort(bottom[:, 0])]

    # Combinare i punti nell'ordine desiderato
    result = np.vstack((odered_top[0], odered_top[1], ordered_bottom[1], ordered_bottom[0]))
    return result

def filter_boxes(boxes, minAR, maxAR,totalArea):
    for index,box in enumerate(boxes):
        #take box width
        w = box[1][0]
        h = box[1][1]
        ratio_1 = w/h
        ratio_2 = h/w
        area = w*h
        print("area")
        print(area)
        print("total area")
        print(totalArea)
        print("ratio 1")
        print(ratio_1)
        print("ratio 2")
        print(ratio_2)

        if (((ratio_1 <= maxAR and ratio_1>=minAR) or (ratio_2 <= maxAR and ratio_2>=minAR)) and area >= totalArea*0.3):
            plate_box = box
            return plate_box, index

def find_image_boxes(contours):
    total_boxes = []
    total_boxesPts = []
    for cnt in contours:
        box = cv2.minAreaRect(cnt)
        boxPts = np.intp(cv2.boxPoints(box))
        total_boxes.append(box)
        total_boxesPts.append(boxPts)
    return total_boxes, total_boxesPts

def find_plate_contours(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.threshold(grey_image, 0, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_image_cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, squareKern)
    binary_image_cleaned = cv2.morphologyEx(binary_image_cleaned, cv2.MORPH_CLOSE, squareKern)

    contours, hierarchy = cv2.findContours(binary_image_cleaned.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    boxes, boxesPts = find_image_boxes(contours)
    tot_area = image.shape[0] * image.shape[1]
    minAR = 0.25
    maxAR = 5
    interested_roi_box, index = filter_boxes(boxes, minAR, maxAR, tot_area)

    print("index found",index)

    interested_roi_boxPts = sortPoint(boxesPts[index])


    return interested_roi_box, interested_roi_boxPts


def transform_image(image):
    roi_contours, roi_contours_pts = find_plate_contours(image)
    warped_detection_image = fourPointsTransform(image, roi_contours_pts)
    return  warped_detection_image
