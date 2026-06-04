from ultralytics import YOLO
import cv2
import math
import pandas as pd

# carregar modelo pose
model = YOLO("yolo11n-pose.pt")

video_path = r"C:\Users\joaov\Desktop\TFM\Videos_TFM\RawVideos\prueba2_left.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

data = []
frame_idx = 0

#a-hip b-knee c-anke NTENHOCRTZVERI
def angle(a, b, c):

    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])

    norm_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    norm_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    # evitar divisão por zero
    if norm_ba == 0 or norm_bc == 0:
        return None

    cos_angle = (
        ba[0]*bc[0] + ba[1]*bc[1]
    ) / (norm_ba * norm_bc)

    # força intervalo [-1,1]
    cos_angle = max(-1, min(1, cos_angle))

    return math.degrees(math.acos(cos_angle))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    if results[0].keypoints is not None:    #modelo encontra pontos numa pessoa->continue
        kpts = results[0].keypoints.xy[0].cpu().numpy()

        # YOLO pose indices:
        # 12 = left hip, 14 = left knee, 16 = left ankle
        # 11 = right hip, 13 = right knee, 15 = right ankle

        hip = kpts[12]   # RIGHT HIP
        knee = kpts[14]  # RIGHT KNEE
        ankle = kpts[16] # RIGHT ANKLE

        ang = angle(hip, knee, ankle)

        if ang is not None:

            time_sec = round(frame_idx / fps, 3)

            data.append([time_sec, ang])

    frame_idx += 1  #passa ao prox frame

cap.release()

# guardar Excel
df = pd.DataFrame(data, columns=["time_sec", "knee_angle"])
output_file = r"C:\Users\joaov\Desktop\TFM\YOLOAnguloJoelhoPrueba2Left.xlsx"
df.to_excel(output_file, index=False)

print("GUARDADO:", output_file)