# import csv
# import copy
# import argparse
# import itertools
# from collections import Counter
# from collections import deque

import cv2 
import numpy as np
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def BoundingBox(image, landmarks, f_width, f_height, alpha):
    image_overlay = image.copy()
    landmark_xs = []
    landmark_ys = []
    for data_point in landmarks.landmark:
        landmark_xs.append(data_point.x)
        landmark_ys.append(data_point.y)

    x1 = int((min(landmark_xs) - 0.01) * f_width)
    x2 = int((max(landmark_xs) + 0.01) * f_width)
    y1 = int((max(landmark_ys) + 0.02) * f_height)
    y2 = int((min(landmark_ys) - 0.02) * f_height)

    pt1 = [x1, y1]
    pt2 = [x2, y2]
    pts = np.array([pt1,pt2], dtype=int)

    cv2.rectangle(
        image_overlay,
        (pts[0][0], pts[0][1]),
        (pts[1][0], pts[1][1]),
        (50, 50, 50),
        -1)
    
    cv2.rectangle(
        image_overlay,
        (pts[0][0], pts[0][1]),
        (pts[1][0], pts[1][1]),
        (200, 200, 50),
        4)
    
    image = cv2.addWeighted(image_overlay, alpha, image, 1 - alpha, 0)
    return image, pts

def BoundingLabel(image, pts, hand, label, f_width):
    image_overlay = image.copy()
    cv2.rectangle(
        image_overlay,
        (pts[1][0], pts[1][1] - 50),
        (pts[0][0], pts[1][1]),
        (200, 200, 50),
        -1)
    image = cv2.addWeighted(image_overlay, alpha + 0.2, image, 0.8 - alpha, 0)
    pts_flip = np.transpose(np.array([int(f_width) - pts[:,0], pts[:,1]], dtype=int))
    image = cv2.flip(image, 1)
    cv2.putText(
        image,
        f'{hand}: {label}',
        (pts_flip[1][0] + 15, pts_flip[1][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        5)
    image = cv2.flip(image, 1)
    return image


def DrawWindowInfo(image, fps, rgest, lgest, gest):
    cv2.putText(
        image,
        'FPS: ' + fps,
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        6)
    if gest != '':
            cv2.putText(
        image,
        'Gesture: ' + gest,
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        6)
    elif lgest != '' or rgest != '':
        if rgest != '':
            cv2.putText(
                image,
                'Right: ' + rgest,
                (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                6)
            if lgest != '':
                cv2.putText(
                    image,
                    'Left: ' + lgest,
                    (25, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 0),
                    6)
        elif lgest != '':
            cv2.putText(
                image,
                'Left: ' + lgest,
                (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                6)
    return image


def CalcFPS(prev_time):

    new_time = time.time() # take time after processing frame in seconds
    fps = 1 / (new_time - prev_time) # calculate for 1 frame between new_time and prev_time
    
    fps = round(fps)
    fps = str(fps) # make it a string so the puText funciton doesn't swear at me

    prev_time = new_time # set prev_time to current new_time for next frame
    return fps, prev_time

# Setting up ideo capture
cam = cv2.VideoCapture(0) # Camera!!
f_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT) # frame height
f_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH) # frame weight
alpha = 0.35 # overlay transparency facotr
new_time = 0
prev_time = 0
# text_y = int(f_height * 1 / 10)
# text_x = int(f_width / 2)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    
    while cam.isOpened():
        success, image = cam.read()

        if not success:
            print('fuck')
            continue

        # marking the image as not writeable to pass by reference and improve performance
        image.flags.writeable = False

        # convert image to rGB and process
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw annotations
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            i = 0 ## poor man's counter frfr 
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                image, pts = BoundingBox(image, hand_landmarks, f_width, f_height, alpha)

                handedness_label = []
                for hand_handedness in results.multi_handedness:
                    ## if statements to flip the labels so it's correct in selfie view
                    if hand_handedness.classification[0].label == 'Right':
                        handedness_label.append('Left')
                    elif hand_handedness.classification[0].label == 'Left':
                        handedness_label.append('Right')
                    else:
                        print('how tf you got a rieft hand?')
                print('')
                print(hand_landmarks, type(hand_landmarks))
                print('')

                hand = handedness_label[i]
                label = 'Filler'
                image = BoundingLabel(image, pts, hand, label, f_width)
                i += 1

        # temp bummy variables:
        rgest = 'Filler'
        lgest = 'Filler'
        gest = ''

        fps, prev_time = CalcFPS(prev_time) # Calc FPS
        image = cv2.flip(image, 1)
        image = DrawWindowInfo(image, fps, rgest, lgest, gest)
        image = cv2.flip(image, 1)
        
        # flip image over y for selfie_view
        cv2.imshow(
            'ASL Interpretation',
            cv2.flip(image, 1))
        
        # if ESC is pressed exit loop/close cam
        if cv2.waitKey(5) & 0xFF == 27:
            break

# release camera
cam.release()
