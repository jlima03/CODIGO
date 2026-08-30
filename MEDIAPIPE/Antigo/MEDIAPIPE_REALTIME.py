import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import math

# ---------------- MediaPipe ----------------

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose()

video_path = r"C:\Users\joaov\Desktop\TFM\Videos_TFM\RawVideos\prueba8_left.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

# ---------------- gráfico ----------------

plt.ion()

fig, ax = plt.subplots()

x_data=[]
y_data=[]

line, = ax.plot([],[])

ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Ângulo do joelho (°)")
ax.set_title("Knee Angle - MediaPipe")

# ---------------- função ângulo ----------------

def angle(a,b,c):

    ba=(a[0]-b[0],a[1]-b[1])
    bc=(c[0]-b[0],c[1]-b[1])

    cos_angle=(ba[0]*bc[0]+ba[1]*bc[1])/(
        math.sqrt(ba[0]**2+ba[1]**2)*
        math.sqrt(bc[0]**2+bc[1]**2)
    )

    cos_angle=max(-1,min(1,cos_angle))

    return math.degrees(math.acos(cos_angle))

frame_idx=0

while cap.isOpened():

    ret,frame=cap.read()

    if not ret:
        break

    h,w,_=frame.shape

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results=pose.process(rgb)

    if results.pose_landmarks:

        # desenhar skeleton
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        lm=results.pose_landmarks.landmark

        # RIGHT LEG
        hip=lm[mp_pose.PoseLandmark.RIGHT_HIP]
        knee=lm[mp_pose.PoseLandmark.RIGHT_KNEE]
        ankle=lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

        hip=(hip.x*w,hip.y*h)
        knee=(knee.x*w,knee.y*h)
        ankle=(ankle.x*w,ankle.y*h)

        ang=angle(hip,knee,ankle)

        t=frame_idx/fps

        # atualizar gráfico
        x_data.append(t)
        y_data.append(ang)

        line.set_xdata(x_data)
        line.set_ydata(y_data)

        ax.relim()
        ax.autoscale_view()

        plt.pause(0.001)

        # mostrar ângulo no vídeo
        #cv2.putText(
        #    frame,
        #    f"{ang:.1f} deg",
        #    (int(knee[0]),int(knee[1])),
        #    cv2.FONT_HERSHEY_SIMPLEX,
        #    1,
        #    (0,255,0),
        #    2
        #)

    cv2.imshow("MediaPipe Skeleton",frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

    frame_idx+=1

cap.release()

cv2.destroyAllWindows()
plt.close()