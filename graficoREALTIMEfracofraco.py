import cv2
import pandas as pd
import matplotlib.pyplot as plt

# carregar dados
df = pd.read_csv(r"C:\Users\joaov\Desktop\TFM\knee_angles.csv")

time = df["time_sec"].values
angle = df["knee_angle"].values

# vídeo
video_path = r"C:\Users\joaov\Desktop\TFM\Videos_TFM\RawVideos\prueba8_left.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

# gráfico
plt.ion()
fig, ax = plt.subplots()

ax.plot(time, angle, label="Knee angle")
marker, = ax.plot([], [], "ro")  # ponto vermelho

ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Ângulo (°)")
ax.set_title("Ângulo do joelho ao longo do tempo")
ax.legend()

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t = frame_idx / fps

    # encontrar ponto mais próximo no tempo
    idx = (abs(time - t)).argmin()

    # atualizar marcador
    marker.set_data([time[idx]], [angle[idx]])

    plt.pause(0.001)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()