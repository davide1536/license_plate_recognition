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


def findObjectsInFrame(frame, outs, objectnessThreshold=0.5, confThreshold = 0.5, nmsThreshold = 0.3, debug = True):
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
