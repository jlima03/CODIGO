import cv2
import mediapipe as mp
import math
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ---------------- ANGLE ----------------

def angle(hip, knee, ankle):
    ba = (hip[0] - knee[0], hip[1] - knee[1])
    bc = (ankle[0] - knee[0], ankle[1] - knee[1])

    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = (math.sqrt(ba[0]**2 + ba[1]**2) *
           math.sqrt(bc[0]**2 + bc[1]**2))

    if mag == 0:
        return None

    cos_angle = dot / mag
    cos_angle = max(-1, min(1, cos_angle))

    return math.degrees(math.acos(cos_angle))




# ---------------- SETUP ----------------

mp_pose = mp.solutions.pose

video_path = r"C:\Users\joaov\Desktop\TFM\Videos_TFM\RawVideos\prueba8_left.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

data = []




#Filtro Butterworth (igual tfm manuel)

def butter_lowpass_filter(data, cutoff=6, fs=30, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# ---------------- PROCESS ----------------

with mp_pose.Pose() as pose:

    frame_idx = 0
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:

            lm = results.pose_landmarks.landmark

            # pontos
            RH = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            RK = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            RA = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

        
            LA = lm[mp_pose.PoseLandmark.LEFT_ANKLE]

            hip = (RH.x, RH.y)
            knee = (RK.x, RK.y)
            ankle = (RA.x, RA.y)

            # ---------------- ANGLE ----------------
            ang = angle(hip, knee, ankle)

            # ---------------- STRIDE DETECTION ----------------
            # diferença entre tornozelos(nao distancia)
            phase = RA.x - LA.x


            
            

            #DISTANCIA TOBILHOS
            ankle_dist = math.sqrt(
                (RA.x - LA.x)**2 + (RA.y - LA.y)**2
            )
            # ---------------- TIME ----------------
            time_sec = frame_idx / fps

            # ---------------- STORE ----------------
            data.append([time_sec, ang, ankle_dist])


        frame_idx += 1  #prox frame

cap.release()



# ---------------- GUARDAR DADOS (EXCEL) ----------------

df = pd.DataFrame(data, columns=["time_sec", "knee_angle", "ankle_raw"])


df["knee_smooth"] = butter_lowpass_filter(df["knee_angle"])
df["ankle_smooth"] = butter_lowpass_filter(df["ankle_raw"])

df.to_excel(
    r"C:\Users\joaov\Desktop\TFM\ComFiltro2.xlsx",
    index=False
)

#-----------------------criar e visualizar maximos
peaks, _ = find_peaks(
    df["ankle_smooth"],
    distance=int(fps * 0.7),   # evita 2 picos na mesma zancada
    prominence=0.01            # ignora ruído pequeno
)

df["is_peak"] = 0
df.loc[peaks, "is_peak"] = 1

#----------------------------------PLOTS

plt.figure(figsize=(12,5))

plt.plot(df["time_sec"], df["ankle_smooth"], label="Ankle smooth")

plt.scatter(
    df["time_sec"][peaks],
    df["ankle_smooth"][peaks],
    color="red",
    label="Peaks"
)

plt.xlabel("Time (s)")
plt.ylabel("Ankle distance")
plt.legend()
plt.tight_layout()
plt.xlim(0, 10)  # mostra só os primeiros 10 segundos
plt.savefig(r"C:\Users\joaov\Desktop\TFM\FotoTeste1.png")
plt.show()
plt.close()




print("Done")