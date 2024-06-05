import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

# Eye aspect ratio thresholds and counters
EAR_THRESH = 0.25
CLOSED_EYE_FRAME_THRESH = 3  # In seconds
closed_eye_counter = 0
start_closed_eye_time = None

# Mouth aspect ratio threshold
MAR_THRESH = 0.6
YAWN_FRAME_THRESH = 3  # In seconds
start_yawn_time = None

def aspect_ratio(pts):
    # Compute the euclidean distances between the two sets of vertical landmarks
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])

    # Compute the euclidean distance between the horizontal landmarks
    C = np.linalg.norm(pts[0] - pts[3])

    # Compute the aspect ratio
    ar = (A + B) / (2.0 * C)

    return ar

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    start = time.time()

    cv2.imshow("Raw", image)

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define indexes for the landmarks of interest
            landmarks_of_interest = [33, 263, 1, 61, 291, 199]
            left_eye_landmarks = [362, 385, 387, 263, 373, 380]
            right_eye_landmarks = [33, 160, 158, 133, 153, 144]
            mouth_landmarks = [61, 81, 311, 191, 78, 95]

            # Get the coordinates of the landmarks
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in landmarks_of_interest:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])
            
            # Eye tracking
            left_eye = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in left_eye_landmarks]
            right_eye = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in right_eye_landmarks]

            # Calculate the eye aspect ratio (EAR) for both eyes
            left_ear = aspect_ratio(np.array(left_eye))
            right_ear = aspect_ratio(np.array(right_eye))

            # Average the EAR for both eyes
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESH:
                if start_closed_eye_time is None:
                    start_closed_eye_time = time.time()
                else:
                    elapsed_time = time.time() - start_closed_eye_time
                    if elapsed_time > CLOSED_EYE_FRAME_THRESH:
                        cv2.putText(image, "ALERT! Eyes Closed", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            else:
                start_closed_eye_time = None

            # Mouth tracking
            mouth = [(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in mouth_landmarks]

            # Calculate the mouth aspect ratio (MAR)
            mar = aspect_ratio(np.array(mouth))

            if mar > MAR_THRESH:
                if start_yawn_time is None:
                    start_yawn_time = time.time()
                else:
                    elapsed_time = time.time() - start_yawn_time
                    if elapsed_time > YAWN_FRAME_THRESH:
                        cv2.putText(image, "ALERT! Yawning", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    else:
                        cv2.putText(image, "ALERT", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            else:
                start_yawn_time = None
                

            # Convert face_2d and face_3d to NumPy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head is tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            end = time.time()
            totalTime = end - start
            fps = 60

            cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    cv2.imshow('Head Pose and Eye Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
