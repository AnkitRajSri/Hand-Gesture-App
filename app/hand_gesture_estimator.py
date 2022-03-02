import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Key, Controller

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
landmark_joints = [[8, 5, 0]]


video = cv2.VideoCapture(0)

with mp_hand.Hands(min_detection_confidence=0.8,
                   min_tracking_confidence=0.5) as hands:
    while video.isOpened():
        ret, frame = video.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = cv2.flip(image, 1)
        image.flags.writeable = False

        result = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for idx, hand_lm in enumerate(result.multi_hand_landmarks):
                mp_draw.draw_landmarks(image, hand_lm, mp_hand.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                                       )
                for joint in landmark_joints:
                    a = np.array([hand_lm.landmark[joint[0]].x, hand_lm.landmark[joint[0]].y])
                    b = np.array([hand_lm.landmark[joint[1]].x, hand_lm.landmark[joint[1]].y])
                    c = np.array([hand_lm.landmark[joint[2]].x, hand_lm.landmark[joint[2]].y])

                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians * 180.0 / np.pi)
                    cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                keyboard = Controller()
                if angle <= 180:
                    keyboard.press(Key.right)
                    keyboard.release(Key.right)
                else:
                    keyboard.press(Key.left)
                    keyboard.release(Key.left)

                cv2.rectangle(image, (0, 0), (355, 73), (214, 44, 53))
                cv2.putText(image, "Direction", (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, "Left" if angle > 180 else "Right",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Tracking", image)
        k = cv2.waitKey(1)

        if k == ord("q"):
            break

video.release()
cv2.destroyAllWindow()
