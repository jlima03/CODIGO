import cv2
import mediapipe as mp
import math
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ---------------- ANGLE ----------------

def angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

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




#Filtro Butterworth 

def butter_lowpass_filter(data, cutoff=6, fs=60, order=4):
    nyq = 0.5 * fs  #freq de nyquist (0.5*60=30hz)////divides por dois pelo teorema de nyquist
    normal_cutoff = cutoff / nyq    #=6hz/15hz=0.2

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
            RF = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            RS = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            LA = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            

            hip = (RH.x, RH.y)
            knee = (RK.x, RK.y)
            ankle = (RA.x, RA.y)
            foot = (RF.x, RF.y)
            shoulder = (RS.x, RS.y)

            # CALCuLAR angulo joelho ----------------
            #ang = angle(hip, knee, ankle)

            # CALCULAR angulo tornozelo ----------------
            #ang = angle(knee, ankle, foot)

            # CALCULAR angulo anca ----------------
            ang = angle(shoulder, hip, knee)

            ang=180-ang

            #DISTANCIA TOBILHOS
            #ankle_dist = math.sqrt(
            #    (RA.x - LA.x)**2 + (RA.y - LA.y)**2 #formula distancia euclidiana: d = RAIZ[(x2-x1)^2 + (y2-y1)^2]
            #)
            ankle_dist = abs(RA.x - LA.x)
            # calc tempo ----------------
            time_sec = frame_idx / fps

            # guardar estes dados ----------------
            data.append([time_sec, ang, ankle_dist])


        frame_idx += 1  #prox frame

cap.release()



# criar dataframe(tabela) ----------------

df = pd.DataFrame(data, columns=["time_sec", "angle", "ankle_dist"])

#aplicar filtro
df["angle_smooth"] = butter_lowpass_filter(df["angle"])
df["angle_raw"] = df["angle"]
df["ankle_dist_smooth"] = butter_lowpass_filter(df["ankle_dist"])




#-----------------------criar e visualizar maximos
peaks, _ = find_peaks(
    df["ankle_dist_smooth"],
    distance=int(fps * 0.7),   # evita 2 picos na mesma zancada(faz q seja de 0.7 em 0.7 secs OU a cada 42 frames( 60*0.7=42 frames)
    prominence=0.01            # ignora ruído pequeno
)




cycles = []
cycles_raw = []


#funcao normalizacao de 101 pontos

def normalize(signal, n=101):    #normalizar p 101 pontos
    x = np.linspace(0, 1, len(signal))#cria um eixo "falso" com os x frames usados na zancada (ex.:41fps 41 pontos no eixo x)
    f = interp1d(x, signal, kind="linear")  #cria uma funcao linear na funcao dada
    x_new = np.linspace(0, 1, n)    #novo eixo com 101 pontos baseado na funcal anterior
    return f(x_new)

for i in range(len(peaks) - 1):

    start = peaks[i]
    end = peaks[i + 1]  #funcao basica de definicao de inicio e final de cada zancada(pico e pico+1)

    stride = df["angle_smooth"].iloc[start:end].values
    

    if len(stride) < 10:   #filtrar zancadas mt pequenas (invalidas)
        continue

    norm = normalize(stride)
    cycles.append(norm)   #guardar todas as zancadas (normalizadas)

    #SEM FILTRO------------

    stride_raw = df["angle_raw"].iloc[start:end].values

    if len(stride_raw) < 10:
        continue

    norm_raw = normalize(stride_raw)
    cycles_raw.append(norm_raw)

cycles = np.array(cycles)

mean = np.mean(cycles, axis=0)    #media c filtro
std = np.std(cycles, axis=0)  #desvio padrao c filtro


cycles_raw = np.array(cycles_raw)

mean_raw = np.mean(cycles_raw, axis=0)  #media s filtro
std_raw = np.std(cycles_raw, axis=0)    #desvio padrao s filtro

df["is_peak"] = 0
df.loc[peaks, "is_peak"] = 1



#PLOT FILTRO E SEM FILTRO-----------------------


x = np.linspace(0, 100, 101)

plt.figure(figsize=(10,5))

# FILTRADO
plt.plot(x, mean, label="Filtered", color="blue")
plt.fill_between(x, mean-std, mean+std, alpha=0.2, color="blue")

# RAW (sem filtro)
plt.plot(x, mean_raw, label="Raw", color="green")
plt.fill_between(x, mean_raw-std_raw, mean_raw+std_raw, alpha=0.2, color="green")

plt.xlabel("% gait cycle")
plt.ylabel("Angle (º)")
plt.legend()
plt.grid()
plt.show()

#PLOT SEM FILTRO
plt.figure(figsize=(10,5))

plt.plot(x, mean_raw, label="Raw", color="green")
plt.fill_between(x, mean_raw-std_raw, mean_raw+std_raw, alpha=0.2, color="green")

plt.xlabel("% gait cycle")
plt.ylabel("Angle (º)")
plt.title("Raw Signal")
plt.legend()
plt.grid()
plt.show()

#PLOT C FILTRO
plt.figure(figsize=(10,5))

plt.plot(x, mean, label="Filtered", color="blue")
plt.fill_between(x, mean-std, mean+std, alpha=0.2, color="blue")

plt.xlabel("% gait cycle")
plt.ylabel("Angle (º)")
plt.title("Filtered Signal")
plt.legend()
plt.grid()
plt.show()


# ANKLE DIST FILTRO VS S FILTRO


plt.figure(figsize=(10,5))

plt.plot(df["time_sec"], df["ankle_dist"], label="Raw", color="green")
plt.plot(df["time_sec"], df["ankle_dist_smooth"], label="Filtered", color="blue")

plt.xlabel("Time (s)")
plt.ylabel("Horizontal ankle distance")
plt.title("Ankle Distance - Raw vs Filtered")
plt.legend()
plt.grid()
plt.show()



# ANKLE DIST S FILTRO


plt.figure(figsize=(10,5))

plt.plot(df["time_sec"], df["ankle_dist"], label="Raw", color="green")

plt.xlabel("Time (s)")
plt.ylabel("Horizontal ankle distance")
plt.title("Ankle Distance - Raw")
plt.legend()
plt.grid()
plt.show()



# ANKLE DIST C FILTRO


plt.figure(figsize=(10,5))

plt.plot(df["time_sec"], df["ankle_dist_smooth"], label="Filtered", color="blue")

plt.xlabel("Time (s)")
plt.ylabel("Horizontal ankle distance")
plt.title("Ankle Distance - Filtered")
plt.legend()
plt.grid()
plt.show()