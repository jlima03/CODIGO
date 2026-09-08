import cv2
import mediapipe as mp
from ultralytics import YOLO
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

video_path = r"C:\Users\joaov\Desktop\TFM\Videos_TFM\RawVideos\prueba2L.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

data = []
data_yolo = []

model = YOLO("yolo11n-pose.pt")




#Filtro Butterworth

def butter_lowpass_filter(data, cutoff=6, fs=60, order=4):          #fs = 17 para prueba 3
    nyq = 0.5 * fs  #freq de nyquist (0.5*60=30hz)////divides por dois pelo teorema de nyquist
    normal_cutoff = cutoff / nyq    #=6hz/30hz=0.2
 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Processamento do video ----------------

with mp_pose.Pose() as pose:

    frame_idx = 0
    

    while True: #Percorre todos os frames
        ret, frame = cap.read() #ler frame
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))#pose.process obtem keypoints
        yolo_results = model(frame, verbose=False)
        yolo_result = yolo_results[0]

        if yolo_result.keypoints is not None and len(yolo_result.keypoints) > 0:

            kp = yolo_result.keypoints.xyn[0].cpu().numpy()

            # Extrair estes pontos (articulacoes)
                        
            RH = kp[12]   # Right Hip
            RK = kp[14]   # Right Knee
            RA = kp[16]   # Right Ankle
            RS = kp[6]    # Right Shoulder
            LH = kp[11]   # Left Hip
            LK = kp[13]   # Left Knee
            LA = kp[15]   # Left Ankle
            LS = kp[5]    # Left Shoulder

            right_hip = tuple(RH)
            right_knee = tuple(RK)
            right_ankle = tuple(RA)
            right_shoulder = tuple(RS)
            left_hip = tuple(LH)
            left_knee = tuple(LK)
            left_ankle = tuple(LA)
            left_shoulder = tuple(LS)

            
            # CALCuLAR angulo right knee ----------------
            #ang_yolo = angle(right_hip, right_knee, right_ankle)
            
            # CALCuLAR angulo left knee ----------------
            #ang_yolo = angle(left_hip, left_knee, left_ankle)
            
            
            # CALCULAR angulo right ankle ----------------       # No HAY ANKLE EM YOLO
            #ang_yolo = angle(right_knee, right_ankle, right_foot)
            
            # CALCULAR angulo left ankle ----------------        # No HAY ANKLE EM YOLO
            #ang_yolo = angle(left_knee, left_ankle, left_foot)
            
            
            # CALCULAR angulo right hip ----------------
            #ang_yolo = angle(right_shoulder, right_hip, right_knee)
            
            # CALCULAR angulo left hip ----------------
            ang_yolo = angle(left_shoulder, left_hip, left_knee)
            
                
            

            ang_yolo = 180 - ang_yolo   # # Ajuste de la convención angular para mantener la coherencia con el TFM de referencia y la literatura consultada

            # distância horizontal entre tornozelos
            ankle_dist_yolo = abs(RA[0] - LA[0])

            time_sec_yolo = frame_idx / fps

            data_yolo.append([time_sec_yolo,ang_yolo,ankle_dist_yolo])

        if results.pose_landmarks:

            lm = results.pose_landmarks.landmark

            # Extrair estes pontos (articulacoes)
            RH = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            RK = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            RA = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
            RF = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            RS = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            LH = lm[mp_pose.PoseLandmark.LEFT_HIP]
            LK = lm[mp_pose.PoseLandmark.LEFT_KNEE]
            LA = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            LF = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            LS = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]

            
            

            right_hip = (RH.x, RH.y)
            right_knee = (RK.x, RK.y)
            right_ankle = (RA.x, RA.y)
            right_foot = (RF.x, RF.y)
            right_shoulder = (RS.x, RS.y)

            left_hip = (LH.x, LH.y)
            left_knee = (LK.x, LK.y)
            left_ankle = (LA.x, LA.y)
            left_foot = (LF.x, LF.y)
            left_shoulder = (LS.x, LS.y)

             


            # CALCuLAR angulo right knee ----------------
            #ang = angle(right_hip, right_knee, right_ankle)

            # CALCuLAR angulo left knee ----------------
            #ang = angle(left_hip, left_knee, left_ankle)


            # CALCULAR angulo right ankle ----------------       
            #ang = angle(right_knee, right_ankle, right_foot)

            # CALCULAR angulo left ankle ----------------        
            #ang = angle(left_knee, left_ankle, left_foot)


            # CALCULAR angulo right hip ----------------
            #ang = angle(right_shoulder, right_hip, right_knee)

            # CALCULAR angulo left hip ----------------
            ang = angle(left_shoulder, left_hip, left_knee)

            ang=180-ang     # Ajuste de la convención angular para mantener la coherencia con el TFM de referencia y la literatura consultada


            #DISTANCIA TOBILHOS
           # ankle_dist = math.sqrt(
           #     (RA.x - LA.x)**2 + (RA.y - LA.y)**2 #formula distancia euclidiana: d = RAIZ[(x2-x1)^2 + (y2-y1)^2]
           # )
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
df["ankle_dist_smooth"] = butter_lowpass_filter(df["ankle_dist"])

df_yolo = pd.DataFrame(
    data_yolo,
    columns=["time_sec", "angle", "ankle_dist"]
)

df_yolo["angle_smooth"] = butter_lowpass_filter(
    df_yolo["angle"]
)

df_yolo["ankle_dist_smooth"] = butter_lowpass_filter(
    df_yolo["ankle_dist"]
)




#FAZER GRAFICO VICON------

df_vicon = pd.read_excel(
    r"C:\Users\joaov\Desktop\TFM\DadosAnalisados\VICON\CAPTURA02.xlsx",
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
    

    right_knee = (
    df_vicon["RKJC_X"].iloc[i],
    df_vicon["RKJC_Y"].iloc[i],
    df_vicon["RKJC_Z"].iloc[i]
    )

    left_knee = (
    df_vicon["LKJC_X"].iloc[i],
    df_vicon["LKJC_Y"].iloc[i],
    df_vicon["LKJC_Z"].iloc[i]
    )


    right_toe = (
    df_vicon["RTOE_X"].iloc[i],
    df_vicon["RTOE_Y"].iloc[i],
    df_vicon["RTOE_Z"].iloc[i]
    )

    left_toe = (
    df_vicon["LTOE_X"].iloc[i],
    df_vicon["LTOE_Y"].iloc[i],
    df_vicon["LTOE_Z"].iloc[i]
    )

    right_shoulder = (
    df_vicon["RSHO_X"].iloc[i],
    df_vicon["RSHO_Y"].iloc[i],
    df_vicon["RSHO_Z"].iloc[i]
    )

    left_shoulder = (
    df_vicon["LSHO_X"].iloc[i],
    df_vicon["LSHO_Y"].iloc[i],
    df_vicon["LSHO_Z"].iloc[i]
    )

    right_hip = (
    df_vicon["RHJC_X"].iloc[i],
    df_vicon["RHJC_Y"].iloc[i],
    df_vicon["RHJC_Z"].iloc[i]
    )

    left_hip = (
    df_vicon["LHJC_X"].iloc[i],
    df_vicon["LHJC_Y"].iloc[i],
    df_vicon["LHJC_Z"].iloc[i]
    )



    # CALCuLAR angulo right knee ----------------
    #ang = angle3D(right_hip, right_knee, right_ankle)

    # CALCuLAR angulo left knee ----------------
    #ang = angle3D(left_hip, left_knee, left_ankle)


    # CALCULAR angulo right ankle ----------------       
    #ang = angle3D(right_knee, right_ankle, right_toe)

    # CALCULAR angulo left ankle ----------------        
    #ang = angle3D(left_knee, left_ankle, left_toe)


    # CALCULAR angulo right hip ----------------
    #ang = angle3D(right_shoulder, right_hip, right_knee)

    # CALCULAR angulo left hip ----------------
    ang = angle3D(left_shoulder, left_hip, left_knee)

    ang=180-ang     # Ajuste de la convención angular para mantener la coherencia con el TFM de referencia y la literatura consultada
    
    vicon_angles.append(ang)

    #CALCULO DIST ANKLES NO VICON
    ankle_dist = abs(
        right_ankle[0] - left_ankle[0]
    )

    vicon_ankle_dist.append(ankle_dist)




vicon_angles = np.array(vicon_angles)

vicon_ankle_dist = np.array(vicon_ankle_dist)





#vicon_ankle_dist_smooth = butter_lowpass_filter(
#    vicon_ankle_dist,
#    cutoff=6,
#    fs=120,
#    order=4
#)






vicon_peaks, _ = find_peaks(
    #vicon_ankle_dist_smooth,                                
    vicon_ankle_dist,
    distance=int(120 * 0.3),        #seleciona todos os picos
    prominence=0.01
)


vicon_peaks=vicon_peaks[::2]    #primeira perna(direita) // seleciona apenas quando a perna direita esta a frente (tal como verificado no video, o primeiro pico é da perna direita)
#vicon_peaks=vicon_peaks[1::2]    #segunda perna(esquerda)



#funcao normalizacao de 101/51 pontos
def normalize(signal, n=101): #normalizar p 101 ou 51 pontos (como tfm manuel(51 pontos))
    signal = np.array(signal)

    x = np.linspace(0, 1, len(signal))#cria um eixo "falso" com os x frames usados na zancada (ex.:41fps 41 pontos no eixo x)
    f = interp1d(x, signal, kind="linear")#cria uma funcao linear na funcao dada
    x_new = np.linspace(0, 1, n)#novo eixo com 101 pontos baseado na funcal anterior
    return f(x_new)

vicon_cycles = []

for i in range(len(vicon_peaks) - 1):       #tens a lista vicon_peaks ex.: vicon_peaks = [120, 245, 368, 492, 618,...]  //como a lista tem x elementos, fazes -1 para que na ultima zancada o "end" exista, por ex.: 10 picos produzem 9 zancadas
    start = vicon_peaks[i]
    end = vicon_peaks[i + 1]

    stride = vicon_angles[start:end]

    if len(stride) < 10:        #ignora zancadas de menos de 10 frames
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
    prominence=0.005           # ignora ruído pequeno
)


#peaks = peaks[::2]  #primeira perna(direita) // divide todos os picos lidos por 2 (le todas as zancada com a mesma perna, neste caso, direita)
peaks = peaks[1::2]  #segunda perna(esquerda) // adicionar ali o 1 para ler os picos mais altos em vez de os mais baixos

peaks_yolo, _ = find_peaks(
    df_yolo["ankle_dist_smooth"],
    distance=int(fps * 0.3),
    prominence=0.005
)
#peaks_yolo = peaks_yolo[::2]   #primeira perna
peaks_yolo = peaks_yolo[1::2]   #segunda perna




# Grafico ankle distance VICON-----------------
#plt.figure(figsize=(12,5))

#plt.plot(
#    #vicon_ankle_dist_smooth,                 
#    vicon_ankle_dist,
#    label="Vicon ankle distance",
#    color="blue"
#)



#plt.scatter(
#    vicon_peaks,
#    #vicon_ankle_dist_smooth[vicon_peaks],
#    vicon_ankle_dist[vicon_peaks],                
#    color="red",
#    s=50,
#    label="Detected peaks"
#)

#for p in vicon_peaks:
#    plt.axvline(
#        p,
#        color="red",
#        alpha=0.3
#    )

#plt.xlabel("Frame")
#plt.ylabel("Ankle distance")
#plt.title("Vicon gait segmentation")
#plt.legend()
#plt.grid()

#plt.show()
#fim grafico vicon ank dist-----



all_peaks, _ = find_peaks(
    df["ankle_dist_smooth"]
)



#plt.figure(figsize=(12,4))

#dist entre tornozelos
#plt.plot(df["time_sec"], df["ankle_dist_smooth"],label="Ankle distance",color="blue")

#plt.scatter(
#    df["time_sec"][all_peaks],
#    df["ankle_dist_smooth"][all_peaks],
#    color="green",
#    s=20,
#    label="all peaks"
#)

#plt.scatter(
#    df["time_sec"][peaks],
#    df["ankle_dist_smooth"][peaks],
#    color="red",
#    s=50,
#    label="selected peaks"
#)

#plt.legend()
#plt.show()




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


cycles_yolo = []

for i in range(len(peaks_yolo) - 1):

    start = peaks_yolo[i]
    end = peaks_yolo[i + 1]

    stride = df_yolo["angle_smooth"].iloc[start:end].values

    if len(stride) < 10:
        continue

    norm = normalize(stride)

    cycles_yolo.append(norm)

cycles_yolo = np.array(cycles_yolo)

mean_yolo = np.mean(cycles_yolo, axis=0)
std_yolo = np.std(cycles_yolo, axis=0)

df["is_peak"] = 0
df.loc[peaks, "is_peak"] = 1


#TABELA

from scipy.stats import pearsonr

# MAE
mae = np.mean(np.abs(mean - vicon_mean))

# RMSE
rmse = np.sqrt(np.mean((mean - vicon_mean) ** 2))

# Correlação de Pearson
r, _ = pearsonr(mean, vicon_mean)

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"Pearson r: {r:.3f}")



#----------------------------------PLOTS


x = np.linspace(0, 100, 101)

#PLOT NORMAL-----------(SEM ZSCCORE)

plt.figure(figsize=(10,5))

 #MediaPipe
plt.plot(x, mean, label="MediaPipe", color="blue")
plt.fill_between(x, mean-std, mean+std, alpha=0.2, color="blue")

#yolo
plt.plot(x,mean_yolo,label="YOLO", color="green")
plt.fill_between(x,mean_yolo-std_yolo,mean_yolo+std_yolo,alpha=0.2,color="green")

 #Vicon (vermelho por cima)
plt.plot(x, vicon_mean, label="Vicon", color="red")
plt.fill_between(x, vicon_mean-vicon_std, vicon_mean+vicon_std, alpha=0.2, color="red")

plt.xlabel("% gait cycle")
plt.ylabel("Angle (º)")
plt.legend()
plt.grid()
plt.show()

# NORMALIZAÇÃO (Z-score)--Tudo abaixo deste codigo é p ZSCORE ---------------------------------------
mp_mean_z = zscore(mean)
mp_std_z = std / np.std(mean)

vicon_mean_z = zscore(vicon_mean)
vicon_std_z = vicon_std / np.std(vicon_mean)

yolo_mean_z = zscore(mean_yolo)
yolo_std_z = std_yolo / np.std(mean_yolo)

plt.figure(figsize=(10,5))

# MediaPipe
plt.plot(x, mp_mean_z, label="MediaPipe (z-score)", color="blue")
plt.fill_between(x,mp_mean_z - mp_std_z,mp_mean_z + mp_std_z,alpha=0.2,color="blue")

# YOLO
plt.plot(x, yolo_mean_z, label="YOLO (z-score)", color="green")
plt.fill_between(x,yolo_mean_z - yolo_std_z,yolo_mean_z + yolo_std_z,alpha=0.2,color="green")

# Vicon
plt.plot(x, vicon_mean_z, label="Vicon (z-score)", color="red")
plt.fill_between(x,vicon_mean_z - vicon_std_z,vicon_mean_z + vicon_std_z,alpha=0.2,color="red")

plt.xlabel("% gait cycle")
plt.ylabel("Normalized angle (z-score)")
plt.legend()
plt.grid()
plt.show()

print("Done")