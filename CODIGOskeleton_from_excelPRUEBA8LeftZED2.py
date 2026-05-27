
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

df = pd.read_excel(r"C:\Users\joaov\Desktop\TFM\Prueba8LeftZED2.xlsx")

bones = [
    ("head","neck"),
    ("neck","spine_top"),
    ("spine_top","spine_mid"),
    ("spine_mid","spine_bottom"),

    ("spine_top","left_shoulder"),
    ("left_shoulder","left_elbow"),
    ("left_elbow","left_hand"),

    ("spine_top","right_shoulder"),
    ("right_shoulder","right_elbow"),
    ("right_elbow","right_hand"),

    ("spine_bottom","left_hip"),
    ("left_hip","left_knee"),
    ("left_knee","left_ankle"),

    ("spine_bottom","right_hip"),
    ("right_hip","right_knee"),
    ("right_knee","right_ankle"),
]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

def get_xyz(joint, frame):
    x = df[f"joint_position_{joint}.x"][frame]
    y = df[f"joint_position_{joint}.y"][frame]
    z = df[f"joint_position_{joint}.z"][frame]
    return x,y,z

def update(frame):

    ax.clear()

    for a,b in bones:

        x1,y1,z1 = get_xyz(a,frame)
        x2,y2,z2 = get_xyz(b,frame)

        if pd.isna(x1) or pd.isna(x2):
            continue

        ax.plot(
            [x1,x2],
            [y1,y2],
            [z1,z2]
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Frame {frame}")

ani = FuncAnimation(
    fig,
    update,
    frames=len(df),
    interval=40
)
ax.view_init(elev=70, azim=90)#angulo visao (elev=90 para paralelo)
plt.show()
