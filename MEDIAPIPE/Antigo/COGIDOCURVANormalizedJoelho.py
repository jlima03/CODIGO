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




# Abrir video ----------------

mp_pose = mp.solutions.pose

video_path = r"C:\Users\joaov\Desktop\TFM\Videos_TFM\RawVideos\prueba8_left.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

data = []




#Filtro Butterworth (igual tfm manuel)

def butter_lowpass_filter(data, cutoff=6, fs=30, order=4):
    nyq = 0.5 * fs  #freq de nyquist (0.5*30=15hz)////divides por dois pelo teorema de nyquist
    normal_cutoff = cutoff / nyq    #=6hz/15hz=0.4

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Processamento do video ----------------

with mp_pose.Pose() as pose:

    frame_idx = 0
    

    while True: #Percorre todos os frames
        ret, frame = cap.read() #ler frame
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))#pose.process obtem keypoints, segunda parte converter para RGB(o q mediapipe usa)

        if results.pose_landmarks:

            lm = results.pose_landmarks.landmark

            # Extrair estes pontos (articulacoes)
            RH = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            RK = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            RA = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
    
        
            LA = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            

            hip = (RH.x, RH.y)
            knee = (RK.x, RK.y)
            ankle = (RA.x, RA.y)
            

            # CALUCLAR angulo joelho ----------------
            ang = angle(hip, knee, ankle)

            


            #DISTANCIA TOBILHOS
            ankle_dist = math.sqrt(
                (RA.x - LA.x)**2 + (RA.y - LA.y)**2 #formula distancia euclidiana: d = RAIZ[(x2-x1)^2 + (y2-y1)^2]
            )
            # calc tempo ----------------
            time_sec = frame_idx / fps

            # guardar estes dados ----------------
            data.append([time_sec, ang, ankle_dist])


        frame_idx += 1  #prox frame

cap.release()



# criar dataframe(tabela) ----------------

df = pd.DataFrame(data, columns=["time_sec", "knee_angle", "ankle_raw"])

#aplicar filtro
df["knee_smooth"] = butter_lowpass_filter(df["knee_angle"])
df["ankle_smooth"] = butter_lowpass_filter(df["ankle_raw"])

df.to_excel(
    r"C:\Users\joaov\Desktop\TFM\ComFiltro2.xlsx",
    index=False
)

#-----------------------criar e visualizar maximos
peaks, _ = find_peaks(
    df["ankle_smooth"],
    distance=int(fps * 0.7),   # evita 2 picos na mesma zancada(faz q seja de 0.7 em 0.7 secs OU a cada 21 frames( 30*0.7=21 frames)
    prominence=0.01            # ignora ruído pequeno
)

import numpy as np
from scipy.interpolate import interp1d

knee_cycles = []

#funcao normalizacao de 51 pontos

def normalize(signal, n=51):    #normalizar p 51 pontos (como tfm manuel)
    x = np.linspace(0, 1, len(signal))#cria um eixo "falso" com os x frames usados na zancada (ex.:41fps 41 pontos no eixo x)
    f = interp1d(x, signal, kind="linear")  #cria uma funcao linear na funcao dada
    x_new = np.linspace(0, 1, n)    #novo eixo com 51 pontos baseado na funcal anterior
    return f(x_new)

for i in range(len(peaks) - 1):

    start = peaks[i]
    end = peaks[i + 1]  #funcao basica de definicao de inicio e final de cada zancada(pico e pico+1)

    knee_stride = df["knee_smooth"].iloc[start:end].values

    if len(knee_stride) < 10:   #filtrar zancadas mt pequenas (invalidas)
        continue

    knee_norm = normalize(knee_stride)

    knee_cycles.append(knee_norm)   #guardar todas as zancadas (normalizadas)

knee_cycles = np.array(knee_cycles)

knee_mean = np.mean(knee_cycles, axis=0)    #media
knee_std = np.std(knee_cycles, axis=0)  #desvio padrao

df["is_peak"] = 0
df.loc[peaks, "is_peak"] = 1

#----------------------------------PLOTS

#plt.figure(figsize=(12,5))

#plt.plot(df["time_sec"], df["ankle_smooth"], label="Ankle smooth")

#plt.scatter(
#    df["time_sec"][peaks],
#    df["ankle_smooth"][peaks],
#    color="red",
#    label="Peaks"
#)

#plt.xlabel("Time (s)")
#plt.ylabel("Ankle distance")
#plt.legend()
#plt.tight_layout()
#plt.xlim(0, 10)  # mostra só os primeiros 10 segundos
#plt.savefig(r"C:\Users\joaov\Desktop\TFM\FotoBolasVermelhas1.png")
#plt.show()
#lt.close()


plt.figure(figsize=(10,5))

x = np.linspace(0, 100, 51)

plt.plot(x, knee_mean, label="Mean knee", color="blue")     #GRAFICO FINAL
plt.fill_between(
    x,
    knee_mean - knee_std,   #Linha azul central
    knee_mean + knee_std,   #Area azul de cada zancada individual
    alpha=0.2,
    color="blue"
)

plt.xlabel("% gait cycle")
plt.ylabel("Angle (º)")
plt.title("Normalized gait cycles")
plt.legend()
plt.grid()

plt.savefig(r"C:\Users\joaov\Desktop\TFM\knee_normalized.png")
plt.show()

print("Done")