import cv2
import mediapipe as mp
import math
import pandas as pd

# ângulo do tornozelo:
# joelho -> tornozelo -> pé

def angle(knee, ankle, foot):

    ba = (knee[0] - ankle[0], knee[1] - ankle[1])
    bc = (foot[0] - ankle[0], foot[1] - ankle[1])

    cos_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (
        math.sqrt(ba[0]**2 + ba[1]**2) *
        math.sqrt(bc[0]**2 + bc[1]**2)
    )

    cos_angle = max(-1, min(1, cos_angle))

    return math.degrees(math.acos(cos_angle))


mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(
    r"C:\Users\joaov\Desktop\TFM\Videos_TFM\RawVideos\prueba8_left.mp4"
)

fps = cap.get(cv2.CAP_PROP_FPS)

frame_idx = 0

data = []

with mp_pose.Pose() as pose:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = pose.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        if results.pose_landmarks:

            lm = results.pose_landmarks.landmark

            knee = (
                lm[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                lm[mp_pose.PoseLandmark.RIGHT_KNEE].y
            )

            ankle = (
                lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y
            )

            foot = (
                lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
            )

            ang = angle(knee, ankle, foot)

            time_sec = round(frame_idx / fps, 3)

            data.append([time_sec, ang])

        frame_idx += 1

cap.release()

df = pd.DataFrame(
    data,
    columns=["time_s", "ankle_angle_deg"]
)

df.to_excel(
    r"C:\Users\joaov\Desktop\TFM\MEDIAPIPEAnguloTornozeloPrueba8Left.xlsx",
    index=False
)

print("excel guardado")