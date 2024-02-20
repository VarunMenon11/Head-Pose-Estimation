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
        for face_landmarks in result.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark): #landmark:- nose, ears, eyes, mouth
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 or idx == 13 or idx == 14: #index of faces
                    if idx == 1: 
                        nose_2d = (lm.x* img_w, lm.y * img_h)
                        nose_3d = (lm.x*img_w, lm.y * img_h, lm.z * 3000)
                    
                    x, y = int(lm.x * img_w), int(lm.y * img_h) #landmark value is scaled by image width and height
                    
                    #2D Coordinates
                    face_2d.append([x,y])
                    #3D Coordinates
                    face_3d.append([x,y, lm.z])
            
            #Convert it to the Numpy Array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            #Camera Calibration for obtaining intrinsic and distortion parameters
            focal_length = 1*img_w
            cam_matrix = np.array([[focal_length, 0, img_h/2],
                                   [0, focal_length, img_w/2],
                                   [0,0,1]])
            
            #Distortion Matrix
            dist_matrix = np.zeros((4,1), dtype = np.float64)

            #solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix) #rot_vec for point has rotated

            rmat, jac = cv2.Rodrigues(rot_vec)

            # Angels
            rotation = Rotation.from_matrix(rmat)
            angles = rotation.as_euler('xyz', degrees=True)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y < -10:
                text = "Looking left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Foward"
            
            #Display
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1])) #Tip of the nose
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0 , 0), 3)

            cv2.putText(image, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x:" + str(np.round(y,2)), (400,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.putText(image, "y:" + str(np.round(x,2)), (400,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.putText(image, "z:" + str(np.round(z,2)), (400,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        
        end = time.time()
        Totaltime = end-start

        #fps = 1/Totaltime
        #print("FPS: ", fps)

        #cv2.putText(image, f"FPS: {int(fps)}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        mp_drawing.draw_landmarks(
               image = image,
                landmark_list = face_landmarks,
                #connections = face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec = drawing_spec
        )

    
    cv2.imshow("Head Pose Estimation", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()

            






