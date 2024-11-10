import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial.transform import Rotation


#Taking Face_mesh from the Mediapipe...used for detecting the face
face_mesh = mp.solutions.face_mesh
facemesh = face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) #Setting up the detection rate

#This is used to display the detection of the face
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness =  1, circle_radius = 1)

#To Open our webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow('Head Pose Estimation', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Head Pose Estimation', 1200, 800)
while cap.isOpened():
    success, image = cap.read()
    start = time.time()
    
    #Converting the Color Space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    result = facemesh.process(image)
    
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape #Height, Width and No of channels(different color channels)
    face_3d = [] #for 3D reference pose
    face_2d = [] #to map the 3D pose to a image plane to place Facial Landmarks from Mediapipe

    if result.multi_face_landmarks: #Multiple face Landmarks
        print(result.multi_face_landmarks)

