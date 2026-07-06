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


#calculo angulo3d(para vicon)
def angle3D(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1, 1)

    return np.degrees(np.arccos(cos_angle))

# Abrir video ----------------

mp_pose = mp.solutions.pose

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
            ang = angle(hip, knee, ankle)

            # CALCULAR angulo tornozelo ----------------
            #ang = angle(knee, ankle, foot)

            # CALCULAR angulo anca ----------------
            #ang = angle(shoulder, hip, knee)


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
print(fps)


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

for i in range(len(df_vicon)):

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

vicon_angles = np.array(vicon_angles)

vicon_peaks, _ = find_peaks(
    vicon_angles,
    distance=int(120 * 0.7),  # 120 Hz do Vicon
    prominence=0.01
)

def normalize(signal, n=51):
    signal = np.array(signal)

    x = np.linspace(0, 1, len(signal))
    f = interp1d(x, signal, kind="linear")

    x_new = np.linspace(0, 1, n)
    return f(x_new)

vicon_cycles = []

for i in range(len(vicon_peaks) - 1):

    start = vicon_peaks[i]
    end = vicon_peaks[i + 1]

    stride = vicon_angles[start:end]

    if len(stride) < 10:
        continue

    norm = normalize(stride, n=51)

    vicon_cycles.append(norm)

vicon_cycles = np.array(vicon_cycles)


vicon_mean = np.mean(vicon_cycles, axis=0)
vicon_std = np.std(vicon_cycles, axis=0)

#ZSCORE-----
def zscore(x):
    x = np.array(x)
    return (x - np.mean(x)) / np.std(x)


#-----------------------criar e visualizar maximos
#peaks, _ = find_peaks(
#    df["ankle_dist_smooth"],
#    distance=int(fps * 0.7),   # evita 2 picos na mesma zancada(faz q seja de 0.7 em 0.7 secs OU a cada 42 frames( 60*0.7=42 frames)
#    prominence=0.01          # ignora ruído pequeno
#)
 


def merge_close_peaks(signal, peaks, min_distance):
    if len(peaks) == 0:
        return np.array([])

    merged = []
    i = 0

    while i < len(peaks):
        current = peaks[i]

        window = [current]
        j = i + 1

        while j < len(peaks) and (peaks[j] - current) < min_distance:
            window.append(peaks[j])
            j += 1

        best_peak = max(window, key=lambda p: signal[p])
        merged.append(best_peak)

        i = j

    return np.array(merged)


#outro graf
all_peaks, _ = find_peaks(df["ankle_dist_smooth"])
plt.figure(figsize=(12,5))


plt.plot(df["ankle_dist_smooth"], label="smooth")
plt.scatter(
    all_peaks,
    df["ankle_dist_smooth"].iloc[all_peaks],
    color="green",
    s=15
)
all_peaks, _ = find_peaks(
    df["ankle_dist_smooth"],
    distance=int(fps * 0.7),
    prominence=0.01
)

peaks = merge_close_peaks(
    df["ankle_dist_smooth"].values,
    all_peaks,
    min_distance=int(fps * 0.01)  # 0.4–0.6s típico de marcha
)
plt.scatter(
    peaks,
    df["ankle_dist_smooth"].iloc[peaks],
    color="red",
    s=50
)
#grafico linhas
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
# ==========================================================
# VALIDAÇÃO DA SEGMENTAÇÃO DAS ZANCADAS
# ==========================================================

# Número de zancadas a visualizar
n_strides = 10

if len(peaks) > n_strides:

    start_idx = peaks[0]
    end_idx = peaks[n_strides]

    # Dados no intervalo selecionado
    t_plot = df["time_sec"].iloc[start_idx:end_idx]

    ankle_plot = df["ankle_dist_smooth"].iloc[start_idx:end_idx]
    angle_plot = df["angle_smooth"].iloc[start_idx:end_idx]

    fig, ax1 = plt.subplots(figsize=(14,6))

    
    # Eixo Y esquerdo -> distância entre tornozelos
    
    ax1.plot(
        t_plot,
        ankle_plot,
        color="tab:blue",
        linewidth=2,
        label="Ankle distance"
    )

    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Ankle distance", color="tab:blue", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Picos da distância entre tornozelos
    peaks_in_range = peaks[
        (peaks >= start_idx) &
        (peaks <= end_idx)
    ]

    ax1.scatter(
        df["time_sec"].iloc[peaks_in_range],
        df["ankle_dist_smooth"].iloc[peaks_in_range],
        color="red",
        s=60,
        zorder=5,
        label="Detected peaks"
    )

    # Linhas verticais = limites das zancadas
    for p in peaks_in_range:
        ax1.axvline(
            x=df["time_sec"].iloc[p],
            color="red",
            linestyle="--",
            alpha=0.5,
            linewidth=1
        )

    # Eixo Y direito -> ângulo do joelho

    ax2 = ax1.twinx()

    ax2.plot(
        t_plot,
        angle_plot,
        color="tab:green",
        linewidth=2,
        label="Knee angle"
    )

    ax2.set_ylabel(
        "Knee angle (deg)",
        color="tab:green",
        fontsize=12
    )

    ax2.tick_params(
        axis="y",
        labelcolor="tab:green"
    )

    # --------------------------------------------------
    # Legenda conjunta
    # --------------------------------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right"
    )

    plt.title(
        "Validation of gait cycle segmentation",
        fontsize=14,
        fontweight="bold"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()


    plt.show()


# FIM VALIDAÇÃO DA SEGMENTAÇÃO DAS ZANCADAS----
cycles = []

#funcao normalizacao de 51 pontos

def normalize(signal, n=51):    #normalizar p 51 pontos (como tfm manuel)
    x = np.linspace(0, 1, len(signal))#cria um eixo "falso" com os x frames usados na zancada (ex.:41fps 41 pontos no eixo x)
    f = interp1d(x, signal, kind="linear")  #cria uma funcao linear na funcao dada
    x_new = np.linspace(0, 1, n)    #novo eixo com 51 pontos baseado na funcal anterior
    return f(x_new)

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



#ALIGNMENT -----------------------------------------


# garantir que ambos têm mesmo tamanho mínimo
#min_len = min(len(df["angle_smooth"]), len(vicon_smooth))

#media = df["angle_smooth"].values[:min_len]
#vicon = vicon_smooth[:min_len]

# usar peaks já calculados (MediaPipe)
# e criar também peaks no Vicon (MESMO método)
#peaks_vicon, _ = find_peaks(vicon, distance=int(120 * 0.7), prominence=0.01)

# alinhar pelo primeiro pico comum
#shift = peaks_vicon[0] - peaks[0]

# corrigir desalinhamento
#vicon_aligned = np.roll(vicon, -shift)

# cortar zonas inválidas após shift
#valid_len = min(len(media), len(vicon_aligned))

#media = media[:valid_len]
#vicon_aligned = vicon_aligned[:valid_len]

#----------------------------------PLOTS
#GRAFICO DIST ANKLES BOLAS VERMELHAS-------
plt.figure(figsize=(12,5))

plt.plot(df["time_sec"], df["ankle_dist_smooth"], label="Ankle smooth")

plt.scatter(
    df["time_sec"][peaks],
    df["ankle_dist_smooth"][peaks],
    color="red",
    label="Peaks"
)

plt.xlabel("Time (s)")
plt.ylabel("Ankle distance")
plt.legend()
plt.tight_layout()
plt.xlim(0, 20)  # mostra só os primeiros 10 segundos
plt.savefig(r"C:\Users\joaov\Desktop\TFM\FotoBolasVermelhas1.png")
plt.show()
plt.close()


#plt.figure(figsize=(10,5))

#x = np.linspace(0, 100, 51)
#
#plt.plot(x, mean, label="Mean", color="blue")     #GRAFICO FINAL
#plt.fill_between(
#    x,
#    mean - std,   #Linha azul central
#    mean + std,   #Area azul de cada zancada individual
#    alpha=0.2,
#    color="blue"
#)

#plt.xlabel("% gait cycle")
#plt.ylabel("Angle (º)")
#plt.title("Normalized gait cycles")
#plt.legend()
#plt.grid()

#plt.savefig(r"C:\Users\joaov\Desktop\TFM\angle_normalized.png")
#plt.show()



x = np.linspace(0, 100, 51)

#PLOT NORMAL-----------

#plt.figure(figsize=(10,5))

# MediaPipe
#plt.plot(x, mean, label="MediaPipe", color="blue")
#plt.fill_between(x, mean-std, mean+std, alpha=0.2, color="blue")

# Vicon (vermelho por cima)
#plt.plot(x, vicon_mean, label="Vicon", color="red")
#plt.fill_between(x, vicon_mean-vicon_std, vicon_mean+vicon_std, alpha=0.2, color="red")

#plt.xlabel("% gait cycle")
#plt.ylabel("Angle (º)")
#plt.legend()
#plt.grid()
#plt.show()

# NORMALIZAÇÃO (Z-score) ---------------------------------------
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

print("Done")