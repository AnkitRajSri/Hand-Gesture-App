import cv2
import mediapipe as mp
import numpy as np
import autopy

# Object to store live video feed captured by the default device
video = cv2.VideoCapture(0)

# Create an object of mediapipe hands
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Create an object to draw the connection between each finger index
mp_draw = mp.solutions.drawing_utils

# Store the width and height of the screen
width_screen, height_screen = autopy.screen.size()

prev_x, prev_y = 0, 0
cur_x, cur_y = 0, 0


def hand_landmarks(col_img):
    landmark_list = []

    # Process the video frames using the mediapipe hands object
    lm_positions = hands.process(col_img)

    # Track the landmarks
    lm_track = lm_positions.multi_hand_landmarks

    if lm_track:
        for hand in lm_track:
            for idx, lm in enumerate(hand.landmark):
                mp_draw.draw_landmarks(img, hand, mp_hand.HAND_CONNECTIONS)
                h, w, c = img.shape
                center_x, center_y = int(lm.x * w), int(lm.y * h)
                landmark_list.append([idx, center_x, center_y])

    return landmark_list


def finger_movements(landmarks):
    # Store 4 sets of binaries
    finger_tips = []

    # Indexes for tips of each finger
    tip_ids = [4, 8, 12, 16, 20]

    # Check if the thump is up
    if landmarks[tip_ids[0]][1] > landmarks[tip_ids[0]-1][1]:
        finger_tips.append(1)
    else:
        finger_tips.append(0)

    # Check if the other fingers are up
    for i in range(1, 5):
        if landmarks[tip_ids[i]][2] < landmarks[tip_ids[0]-3][2]:
            finger_tips.append(1)
        else:
            finger_tips.append(0)

    return finger_tips


while video.isOpened():
    # Read frames from the video captured
    _, img = video.read()

    # Convert the frames into an RGB image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lm_list = hand_landmarks(img_rgb)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if len(lm_list) != 0:
        x_1, y_1 = lm_list[8][1:]
        x2_2, y_2 = lm_list[12][1:]
        finger_tips = finger_movements(lm_list)

        if finger_tips[1] == 1 and finger_tips[2] == 0:
            x_3 = np.interp(x_1, (75, 640 - 75), (0, width_screen))
            y_3 = np.interp(y_1, (75, 480 - 75), (0, height_screen))

            cur_x = prev_x + (x_3 - prev_x) / 7
            cur_y = prev_y + (y_3 - prev_y) / 7

            autopy.mouse.move(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

        if finger_tips[0] == 1 and finger_tips[1] == 0:
            autopy.mouse.click()
        print(finger_tips)

    cv2.imshow("Hand Gesture", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# video.release()
# cv2.destroyAllWindows()
