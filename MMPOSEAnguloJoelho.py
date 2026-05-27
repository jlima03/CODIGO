import cv2
import math
import pandas as pd
from mmpose.apis import MMPoseInferencer

video_path = r"C:\Users\joaov\Desktop\TFM\Videos_TFM\RawVideos\prueba8_left.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

inferencer = MMPoseInferencer(pose2d='rtmo')  # MAIS LEVE

def angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0]*bc[0] + ba[1]*bc[1]
    norm = (math.sqrt(ba[0]**2 + ba[1]**2) *
            math.sqrt(bc[0]**2 + bc[1]**2))

    if norm == 0:
        return None

    cos_angle = dot / norm
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))

data = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 SKIP FRAMES (3x mais rápido)
    if frame_idx % 3 != 0:
        frame_idx += 1
        continue

    # 🔥 REDUZ RESOLUÇÃO
    frame = cv2.resize(frame, (640, 360))

    # inferência
    result = list(inferencer(frame, show=False))[0]

    if 'predictions' in result and len(result['predictions']) > 0:
        kpts = result['predictions'][0][0]['keypoints']

        hip = kpts[11][:2]
        knee = kpts[12][:2]
        ankle = kpts[13][:2]

        ang = angle(hip, knee, ankle)

        if ang is not None:
            time_sec = frame_idx / fps
            data.append([time_sec, ang])

    frame_idx += 1

cap.release()

df = pd.DataFrame(data, columns=["time_sec", "knee_angle"])
df.to_excel(r"C:\Users\joaov\Desktop\TFM\mmposePrueba8LeftAngulo_Joelho.xlsx", index=False)

print("Excel criado")