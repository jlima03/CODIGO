import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Ler o Excel
df = pd.read_excel(
    r"C:\Users\joaov\Desktop\TFM\DadosAnalisados\VICON\CAPTURA08.xlsx",
    header=3
)

# Criar figura
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")
def get_xyz(marker, frame):
    x = df[f"{marker}_X"].iloc[frame]
    y = df[f"{marker}_Y"].iloc[frame]
    z = df[f"{marker}_Z"].iloc[frame]
    return x, y, z




bones = [

    # Tronco
    ("PELVC", "T10"),
    ("T10", "CTORAX"),
    ("CTORAX", "CLAV"),
    ("CLAV", "C7"),

    # Ombros
    ("CLAV", "LSJC"),
    ("CLAV", "RSJC"),
    ("LSJC", "RSJC"),

    # Braço esquerdo
    ("LSJC", "LEJC"),
    ("LEJC", "LWJC"),

    # Braço direito
    ("RSJC", "REJC"),
    ("REJC", "RWJC"),

    # Bacia
    ("PELVC", "LHJC"),
    ("PELVC", "RHJC"),
    ("LHJC", "RHJC"),

    # Perna esquerda
    ("LHJC", "LKJC"),
    ("LKJC", "LAJC"),
    ("LAJC", "LTOE"),

    # Perna direita
    ("RHJC", "RKJC"),
    ("RKJC", "RAJC"),
    ("RAJC", "RTOE")
]



def update(frame):

    ax.clear()

    for a, b in bones:

        try:
            x1, y1, z1 = get_xyz(a, frame)
            x2, y2, z2 = get_xyz(b, frame)
        except KeyError:
            continue

        if pd.isna(x1) or pd.isna(x2):
            continue

        ax.plot(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color="blue",
            linewidth=2
        )

        ax.scatter(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color="red",
            s=20
        )

    # Ajustar os limites automaticamente
    ax.set_xlim(df.filter(regex="_X$").min().min(),
                df.filter(regex="_X$").max().max())

    ax.set_ylim(df.filter(regex="_Y$").min().min(),
                df.filter(regex="_Y$").max().max())

    ax.set_zlim(df.filter(regex="_Z$").min().min(),
                df.filter(regex="_Z$").max().max())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(f"Frame {frame}")

    ax.view_init(elev=70, azim=90)

ani = FuncAnimation(
    fig,
    update,
    frames=len(df),
    interval=8
)

plt.show()