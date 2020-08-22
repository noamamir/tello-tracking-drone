import numpy as np
import cv2
#from darkflow.net.build import TFnet

network = cv2.dnn.readNetFromDarknet("yolov4-tiny-traffic.cfg", "yolov4-tiny-traffic_final.weights")
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
'''
actionNames = []
with open ("traffic.names", "r") as f:
    actionNames = [line.strip() for line in f.readlines()]
'''

actionNames = ["track","flip","stationary","land"]


layerNames = network.getLayerNames()
outputLayers = [layerNames[i[0] - 1] for i in network.getUnconnectedOutLayers()]

def detectActions(img, currentAction):
    actionToReturn = currentAction

    #load image
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)

    network.setInput(blob)

    outs = network.forward(outputLayers)

    actionsID = []
    confidences = []
    boxes = []

    for detections in outs:
            for output in detections:
                    score = output[5:]
                    actionID = np.argmax(score)
                    confidence = score[actionID]
                    if confidence > 0.6:
                        # action detected
                        print( height)
                        centerX = int(output[0] * width)
                        centerY = int(output[1] * height)

                        w = int(output[2] * width)
                        h = int(output[3] * height)

                        x = int(centerX - w/2)
                        y = int(centerY - h/2)

                        boxes.append([x,y,w,h])
                        confidences.append(float(confidence))
                        actionsID.append(actionID)



    indexes = cv2.dnn.NMSBoxes(boxes,confidences, 0.4, 0.6)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            actionToReturn = str(actionNames[actionsID[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img,actionToReturn,(x,y),font,3,(0,255,0),2)


    return img, actionToReturn





