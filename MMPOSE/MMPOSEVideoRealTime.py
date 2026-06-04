from mmpose.apis import MMPoseInferencer
import cv2
import math
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

# -------------------------
# Função ângulo
# -------------------------

def angle(a, b, c):

    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])

    dot = ba[0]*bc[0] + ba[1]*bc[1]

    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba == 0 or mag_bc == 0:
        return None

    cos_angle = dot/(mag_ba*mag_bc)

    # evitar erros numéricos
    cos_angle=max(-1,min(1,cos_angle))

    return math.degrees(math.acos(cos_angle))


# -------------------------
# carregar inferencer
# -------------------------

inferencer = MMPoseInferencer('human')

video_path=r"C:\Users\joaov\Desktop\TFM\Videos_TFM\RawVideos\prueba8_left.mp4"

cap=cv2.VideoCapture(video_path)

fps=cap.get(cv2.CAP_PROP_FPS)

# -------------------------
# gráfico tempo real
# -------------------------

plt.ion()

fig,ax=plt.subplots()

xdata=deque(maxlen=300)
ydata=deque(maxlen=300)

line,=ax.plot([],[])

ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Ângulo joelho")
ax.set_ylim(0,180)

frame_idx=0

data=[]


while True:

    ret,frame=cap.read()

    if not ret:
        break

    result=next(
        inferencer(frame,show=False)
    )

    predictions=result["predictions"]

    if len(predictions)>0:

        kpts=predictions[0][0]["keypoints"]

        # COCO índices
        # 12 right hip
        # 14 right knee
        # 16 right ankle

        hip=kpts[12]
        knee=kpts[14]
        ankle=kpts[16]

        ang=angle(
            hip,
            knee,
            ankle
        )

        if ang is not None:

            t=frame_idx/fps

            xdata.append(t)
            ydata.append(ang)

            data.append([t,ang])

            line.set_xdata(xdata)
            line.set_ydata(ydata)

            ax.relim()
            ax.autoscale_view()

            plt.draw()
            plt.pause(0.001)

            cv2.putText(
                frame,
                f"Knee:{ang:.1f}",
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

    cv2.imshow(
        "MMPose Skeleton",
        frame
    )

    if cv2.waitKey(1)==27:
        break

    frame_idx+=1


cap.release()
cv2.destroyAllWindows()

df=pd.DataFrame(
    data,
    columns=["time_sec","knee_angle"]
)

df.to_excel(
    r"C:\Users\joaov\Desktop\TFM\MMPoseKnee.xlsx",
    index=False
)

print("Excel guardado")