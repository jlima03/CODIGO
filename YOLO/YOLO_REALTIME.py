from ultralytics import YOLO
import cv2
import math
import matplotlib.pyplot as plt

model = YOLO("yolo11n-pose.pt")

video_path = r"C:\Users\joaov\Desktop\TFM\Videos_TFM\RawVideos\prueba3L.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# ---------------- gráfico ----------------
plt.ion()
fig, ax = plt.subplots()

x_data = []
y_data = []
line, = ax.plot([], [])

ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Ângulo joelho (°)")
ax.set_title("YOLO Pose - Knee Angle Live")

# ---------------- função ângulo ----------------
def angle(a, b, c):
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])

    cos_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (
        math.sqrt(ba[0]**2 + ba[1]**2) *
        math.sqrt(bc[0]**2 + bc[1]**2)
    )

    return math.degrees(math.acos(cos_angle))

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    if results[0].keypoints is not None:
        kpts = results[0].keypoints.xy[0].cpu().numpy()

        hip = kpts[12]
        knee = kpts[14]#right knee
        ankle = kpts[16]

        ang = angle(hip, knee, ankle)

        t = frame_idx / fps

        # -------- atualizar dados --------
        x_data.append(t)
        y_data.append(ang)

        line.set_xdata(x_data)
        line.set_ydata(y_data)

        ax.relim()
        ax.autoscale_view()

        plt.pause(0.001)

    # -------- mostrar vídeo com skeleton --------
    annotated = results[0].plot()
    cv2.imshow("YOLO Pose", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()