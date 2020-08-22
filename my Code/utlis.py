import faceDistance as fd
from djitellopy import Tello
import cv2
import numpy as np
import time
import dlib as dlib
import face_recognition
import cvlib as cv

dlib.DLIB_USE_CUDA = True
print (dlib.cuda.get_num_devices())
print(dlib.DLIB_USE_CUDA)

def initTello():
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0
    print(myDrone.get_battery())
    time.sleep(1)
    myDrone.streamoff()
    myDrone.streamon()


    return myDrone

def resetVelocity(myDrone):
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0

def changeVelocity(myDrone):
    myDrone.send_rc_control(myDrone.left_right_velocity,
                            myDrone.for_back_velocity,
                            myDrone.up_down_velocity,
                            myDrone.yaw_velocity)

def getCameraFeed(myDrone, width , height):
    feedFrame = myDrone.get_frame_read()
    feedFrame = feedFrame.frame
    img = cv2.resize(feedFrame,(width,height))
    return img

def findFace(img, width, height):
    faces, confidences = cv.detect_face(img, enable_gpu=False)

    allFacesLoc = []
    allFacesDistance = []

    for(left, top, right, bottom) in faces:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        centerX = (right + left)/2
        centerY = (top+ bottom)/2

        distance = fd.distance_to_camera(bottom - top)
        allFacesDistance.append(distance)
        allFacesLoc.append([centerX,centerY])

    # if face is found
    if len(allFacesDistance) != 0:
        largestFaceIndex = allFacesDistance.index(min(allFacesDistance))
        cv2.line(img, (int(allFacesLoc[largestFaceIndex][0]), int(allFacesLoc[largestFaceIndex][1])), (width//2, height//2), (0, 255, 0), thickness=2)
        return img, [allFacesLoc[largestFaceIndex], allFacesDistance[largestFaceIndex]]
    # in case there no face recognized
    else:
        return img,[[0,0],0]

# TODO: ADD HEIGHT
def trackFaceAll(myDrone, faceInfo, width, height, pidsYaw, pidsPitch, pidsThrottle, prevErrorYaw, prevErrorPitch, prevErrorThrottle, lastSpeedYaw=0, idealDistance=110):
    # all movement
    # PID controller
    # Basic pid lop
    # pixel to cm ratio = 80/1
    #speed = error*pGain + error*dGain + error*iGain (i gain optinal)

    errorPitch = (faceInfo[1] - idealDistance)
    speedPitch = pidsPitch[0] * errorPitch + pidsPitch[1] * (errorPitch - prevErrorPitch)

    # limit the speed, deg per second
    speedPitch = int(np.clip(speedPitch, -80, 80))

    print(speedPitch)
    errorThrottle = faceInfo[0][1] - (height) // 2
    speedThrottle = pidsThrottle[0] * errorThrottle + pidsThrottle[1] * (errorThrottle - prevErrorThrottle)

    # limit the speed, deg per second
    speedThrottle = -(int(np.clip(speedThrottle, -60, 60)))

    errorYaw = faceInfo[0][0] - (width)//2
    speedYaw = pidsYaw[0]* errorYaw + pidsYaw[1]*(errorYaw - prevErrorYaw)

    #limit the speed, deg per second
    speedYaw = int(np.clip(speedYaw, -55, 55))


    if faceInfo [0][0] != 0:
        myDrone.yaw_velocity = speedYaw
        myDrone.for_back_velocity = speedPitch
        myDrone.up_down_velocity = speedThrottle
        lastSpeedYaw = int(np.clip(speedYaw, -50, 50))
    else:
        resetVelocity(myDrone)
        myDrone.yaw_velocity = lastSpeedYaw
        print(lastSpeedYaw)
        errorPitch = 0
        errorYaw = 0
        errorThrottle = 0

    if myDrone.send_rc_control:
        changeVelocity(myDrone)

    return errorYaw, errorPitch, errorThrottle, lastSpeedYaw

def flipRight(myDrone):
    myDrone.flip_right()

def land(myDrone):
    myDrone.land()

def takeOff(myDrone):
    myDrone.takeoff()

def trackFaceYaw(myDrone, faceInfo, width, pids, prevError, lastSpeedYaw=0):
    #Yaw movement only
    # PID controller
    # Basic pid lop
    #speed = error*pGain + error*dGain + error*iGain (i gain optinal)

    error = faceInfo[0][0] - width//2
    speed = pids[0]* error + pids[1]*(error - prevError)

    # limit the speed, deg per second
    speed = int(np.clip(speed, -55, 55))

    if faceInfo [0][0] != 0:
        myDrone.yaw_velocity = speed
    else:
       resetVelocity(myDrone)
       error = 0

    if myDrone.send_rc_control:
        changeVelocity(myDrone)

    return error

#def initiateCommand(myDrone, command)