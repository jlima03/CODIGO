import cv2
from ultralytics import YOLO
import math
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#carregar modelo pose
model = YOLO("yolo11n-pose.pt")

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


#calculo angulo3d(para vicon)
def angle3D(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1, 1)

    return np.degrees(np.arccos(cos_angle))

# Abrir video ----------------



video_path = r"C:\Users\joaov\Desktop\TFM\Videos_TFM\RawVideos\prueba8_left.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

data = []




#Filtro Butterworth (igual tfm manuel)

def butter_lowpass_filter(data, cutoff=6, fs=60, order=4):
    nyq = 0.5 * fs  #freq de nyquist (0.5*30=15hz)////divides por dois pelo teorema de nyquist
    normal_cutoff = cutoff / nyq    #=6hz/15hz=0.4
 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Processamento do video ----------------


frame_idx = 0
    
plt.ion()

fig, ax = plt.subplots(figsize=(10,4))

x_data = []
y_data = []

line, = ax.plot([], [], 'b-')

ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Ankle distance")
ax.set_title("YOLO - Ankle distance em tempo real")

ax.set_xlim(0, 40)       # duração aproximada do vídeo (ajusta se necessário)
ax.set_ylim(0, 250)      # ajusta conforme o teu vídeo

plt.show(block=False)
while True: #Percorre todos os frames
        ret, frame = cap.read() #ler frame
        if not ret:
            break

        results = model(frame, verbose=False)#obtem keypoints

        result = results[0]

        
        if result.keypoints is not None and len(result.keypoints.xy) > 0:

            kp = result.keypoints.xy[0].cpu().numpy()


            # Extrair estes pontos (articulacoes)
            
            RH = kp[12]
            RK = kp[14]
            RA = kp[16]
            RS = kp[6]
            LA = kp[15]

            
            

            hip = tuple(RH)
            knee = tuple(RK)
            ankle = tuple(RA)
            shoulder = tuple(RS)

            # CALCuLAR angulo joelho ----------------
            ang = angle(hip, knee, ankle)

            # CALCULAR angulo tornozelo ----------------
            #ang = angle(knee, ankle, foot)

            # CALCULAR angulo anca ----------------
            #ang = angle(shoulder, hip, knee)


            #DISTANCIA TOBILHOS
            #ankle_dist = math.sqrt(
            #     (RA[0] - LA[0])**2 + (RA[1] - LA[1])**2 #formula distancia euclidiana: d = RAIZ[(x2-x1)^2 + (y2-y1)^2]
            #)

            ankle_dist = (RA[0] - LA[0])

            # calc tempo ----------------
            time_sec = frame_idx / fps
            
            # atualizar gráfico
            x_data.append(time_sec)
            y_data.append(ankle_dist)

            line.set_data(x_data, y_data)

            ax.set_xlim(0, max(5, time_sec))

            # Atualiza automaticamente o eixo Y
            ax.set_ylim(
                min(y_data)-10,
                max(y_data)+10
            )

            fig.canvas.draw()
            fig.canvas.flush_events()
            

            # guardar estes dados ----------------
            data.append([time_sec, ang, ankle_dist])


        frame_idx += 1  #prox frame
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()



# criar dataframe(tabela) ----------------

df = pd.DataFrame(data, columns=["time_sec", "angle", "ankle_dist"])

#aplicar filtro
df["angle_smooth"] = butter_lowpass_filter(df["angle"])
df["ankle_dist_smooth"] = butter_lowpass_filter(df["ankle_dist"])



#FAZER GRAFICO VICON------

df_vicon = pd.read_excel(
    r"C:\Users\joaov\Desktop\TFM\DadosAnalisados\VICON\CAPTURA08.xlsx",
    header=3
)


vicon_angles = []
vicon_ankle_dist = []

for i in range(len(df_vicon)):

    right_ankle = (
    df_vicon["RAJC_X"].iloc[i],
    df_vicon["RAJC_Y"].iloc[i],
    df_vicon["RAJC_Z"].iloc[i]
)

    left_ankle = (
    df_vicon["LAJC_X"].iloc[i],
    df_vicon["LAJC_Y"].iloc[i],
    df_vicon["LAJC_Z"].iloc[i]
)

    knee = (
    df_vicon["RKJC_X"].iloc[i],
    df_vicon["RKJC_Y"].iloc[i],
    df_vicon["RKJC_Z"].iloc[i]
    )

    ankle = (
    df_vicon["RAJC_X"].iloc[i],
    df_vicon["RAJC_Y"].iloc[i],
    df_vicon["RAJC_Z"].iloc[i]
    )

    toe = (
    df_vicon["RTOE_X"].iloc[i],
    df_vicon["RTOE_Y"].iloc[i],
    df_vicon["RTOE_Z"].iloc[i]
    )

    shoulder = (
    df_vicon["RSHO_X"].iloc[i],
    df_vicon["RSHO_Y"].iloc[i],
    df_vicon["RSHO_Z"].iloc[i]
    )

    hip = (
    df_vicon["RHJC_X"].iloc[i],
    df_vicon["RHJC_Y"].iloc[i],
    df_vicon["RHJC_Z"].iloc[i]
    )


    #CALCULAR ANG HIP VICON
    #ang = angle3D(shoulder, hip, knee)

    #CALCULAR ANG ANKLE VICON
    #ang = angle3D(knee, ankle, toe)

    #CALCULAR ANG KNEE VICON
    ang = angle3D(hip, knee, ankle)
    
    vicon_angles.append(ang)

    #CALCULO DIST ANKLES NO VICON
    ankle_dist = abs(
        right_ankle[0] - left_ankle[0]
    )

    vicon_ankle_dist.append(ankle_dist)

vicon_angles = np.array(vicon_angles)

vicon_ankle_dist = np.array(vicon_ankle_dist)

vicon_ankle_dist_smooth = butter_lowpass_filter(
    vicon_ankle_dist,
    cutoff=6,
    fs=120,
    order=4
)

vicon_peaks, _ = find_peaks(
    vicon_ankle_dist_smooth,
    distance=int(120 * 0.7),
    prominence=0.01
)
#funcao normalizacao de 101/51 pontos

def normalize(signal, n=101): #normalizar p 101 ou 51 pontos (como tfm manuel(51 pontos))
    signal = np.array(signal)

    x = np.linspace(0, 1, len(signal))#cria um eixo "falso" com os x frames usados na zancada (ex.:41fps 41 pontos no eixo x)
    f = interp1d(x, signal, kind="linear")#cria uma funcao linear na funcao dada
    x_new = np.linspace(0, 1, n)#novo eixo com 51 pontos baseado na funcal anterior
    return f(x_new)

vicon_cycles = []

for i in range(len(vicon_peaks) - 1):

    start = vicon_peaks[i]
    end = vicon_peaks[i + 1]

    stride = vicon_angles[start:end]

    if len(stride) < 10:
        continue

    norm = normalize(stride, n=101)

    vicon_cycles.append(norm)

vicon_cycles = np.array(vicon_cycles)


vicon_mean = np.mean(vicon_cycles, axis=0)
vicon_std = np.std(vicon_cycles, axis=0)

#ZSCORE-----
def zscore(x):
    x = np.array(x)
    return (x - np.mean(x)) / np.std(x)



#-----------------------criar e visualizar maximos
peaks, _ = find_peaks(
    df["ankle_dist_smooth"],
    distance=int(fps * 0.3),   # Le todos os picos(todas as zancadas)faz q seja de 0.3 em 0.3 secs OU a cada 18 frames( 60*0.3=18 frames)
    prominence=0.01            # ignora ruído pequeno
)

#peaks = peaks[::2]  #divide todos os picos lidos por 2 (le todas as zancada com a mesma perna, neste caso, direita)
peaks = peaks[1::2] #adicionar ali o 1 para ler os picos mais altos em vez de os mais baixos

# Grafico ankle distance VICON-----------------


plt.figure(figsize=(12,5))

plt.plot(
    vicon_ankle_dist_smooth,
    label="Vicon ankle distance"
)

plt.scatter(
    vicon_peaks,
    vicon_ankle_dist_smooth[vicon_peaks],
    color="red",
    s=50,
    label="Detected peaks"
)

for p in vicon_peaks:
    plt.axvline(
        p,
        color="red",
        alpha=0.3
    )

plt.xlabel("Frame")
plt.ylabel("Ankle distance")
plt.title("Vicon gait segmentation")
plt.legend()
plt.grid()

plt.show()
#fim grafico vicon ank dist-----

#grafico linhas-------------------
plt.figure(figsize=(12,4))

plt.plot(df["time_sec"], df["ankle_dist_smooth"])

plt.scatter(
    df["time_sec"][peaks],
    df["ankle_dist_smooth"][peaks],
    color="red"
)

for p in peaks:
    plt.axvline(df["time_sec"].iloc[p], color="red", alpha=0.3)

plt.show()
#fim grafico linhas-----------

all_peaks, _ = find_peaks(
    df["ankle_dist_smooth"]
)

print(len(all_peaks))

plt.figure(figsize=(12,4))

plt.plot(df["time_sec"], df["ankle_dist_smooth"])

plt.scatter(
    df["time_sec"][all_peaks],
    df["ankle_dist_smooth"][all_peaks],
    color="green",
    s=20,
    label="all peaks"
)

plt.scatter(
    df["time_sec"][peaks],
    df["ankle_dist_smooth"][peaks],
    color="red",
    s=50,
    label="selected peaks"
)

plt.legend()
plt.show()


print(peaks)
print(np.diff(peaks))

cycles = []



for i in range(len(peaks) - 1):

    start = peaks[i]
    end = peaks[i + 1]  #funcao basica de definicao de inicio e final de cada zancada(pico e pico+1)

    stride = df["angle_smooth"].iloc[start:end].values

    if len(stride) < 10:   #filtrar zancadas mt pequenas (invalidas)
        continue

    norm = normalize(stride)

    cycles.append(norm)   #guardar todas as zancadas (normalizadas)

cycles = np.array(cycles)

mean = np.mean(cycles, axis=0)    #media
std = np.std(cycles, axis=0)  #desvio padrao

df["is_peak"] = 0
df.loc[peaks, "is_peak"] = 1



#----------------------------------PLOTS
#GRAFICO DIST ANKLES BOLAS VERMELHAS-------
#plt.figure(figsize=(12,5))

#plt.plot(df["time_sec"], df["ankle_dist_smooth"], label="Ankle smooth")

#plt.scatter(
#    df["time_sec"][peaks],
#    df["ankle_dist_smooth"][peaks],
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
#plt.close()



x = np.linspace(0, 100, 101)

#PLOT NORMAL-----------(SEM ZSCCORE)

#plt.figure(figsize=(10,5))

 #MediaPipe
#plt.plot(x, mean, label="MediaPipe", color="blue")
#plt.fill_between(x, mean-std, mean+std, alpha=0.2, color="blue")

 #Vicon (vermelho por cima)
#plt.plot(x, vicon_mean, label="Vicon", color="red")
#plt.fill_between(x, vicon_mean-vicon_std, vicon_mean+vicon_std, alpha=0.2, color="red")

#plt.xlabel("% gait cycle")
#plt.ylabel("Angle (º)")
#plt.legend()
#plt.grid()
#plt.show()

# NORMALIZAÇÃO (Z-score)--Tudo abaixo deste codigo é p ZSCORE ---------------------------------------
mp_mean_z = zscore(mean)
mp_std_z = std / np.std(mean)

vicon_mean_z = zscore(vicon_mean)
vicon_std_z = vicon_std / np.std(vicon_mean)

plt.figure(figsize=(10,5))

# MediaPipe
plt.plot(x, mp_mean_z, label="MediaPipe (z-score)", color="blue")
plt.fill_between(x,
                 mp_mean_z - mp_std_z,
                 mp_mean_z + mp_std_z,
                 alpha=0.2,
                 color="blue")

# Vicon
plt.plot(x, vicon_mean_z, label="Vicon (z-score)", color="red")
plt.fill_between(x,
                 vicon_mean_z - vicon_std_z,
                 vicon_mean_z + vicon_std_z,
                 alpha=0.2,
                 color="red")

plt.xlabel("% gait cycle")
plt.ylabel("Normalized angle (z-score)")
plt.legend()
plt.grid()
plt.show()

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

print("Done")