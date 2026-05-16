import cv2
import mediapipe as mp
import math
import pandas as pd

def angle(hip, knee, ankle):
    ba = (hip[0] - knee[0], hip[1] - knee[1])
    bc = (ankle[0] - knee[0], ankle[1] - knee[1])

    cos_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (
        math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2)
    )

    return math.degrees(math.acos(cos_angle))


mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(
    r"C:\Users\joaov\Desktop\TFM\Videos_TFM\RawVideos\prueba2_left.mp4"
)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

data = []

with mp_pose.Pose() as pose:

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:

            lm = results.pose_landmarks.landmark

            hip = (
                lm[mp_pose.PoseLandmark.RIGHT_HIP].x,
                lm[mp_pose.PoseLandmark.RIGHT_HIP].y
            )

            knee = (
                lm[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                lm[mp_pose.PoseLandmark.RIGHT_KNEE].y
            )

            ankle = (
                lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y
            )

            ang = angle(hip, knee, ankle)

            time_sec = round(frame_idx / fps, 3)

            data.append([time_sec, ang])

        frame_idx += 1

cap.release()

df = pd.DataFrame(data, columns=["time_s", "angle_deg"])

df.to_excel(
    r"C:\Users\joaov\Desktop\TFM\angles.csv",
    index=False
)

print("Excel guardado")