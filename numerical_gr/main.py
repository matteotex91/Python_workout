import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

NT = 20
NX = 20
NY = 20

MAXT = 1
MAXX = 1
MAXY = 1

k = 1e-9

DT = 2 * MAXT / NT
DX = 2 * MAXX / NX
DY = 2 * MAXY / NY
D_V = [DT, DX, DY]

T_VEC = np.linspace(-MAXT, MAXT, NT)
X_VEC = np.linspace(-MAXX, MAXX, NX)
Y_VEC = np.linspace(-MAXY, MAXY, NY)


g = np.zeros((NT, NX, NY, 3, 3))  # metrics
T = np.zeros_like(g)
R = np.zeros_like(g)
G = np.zeros((NT, NX, NY, 3, 3, 3))  # christoffel symbols


for it in range(NT):
    for ix in range(NX):
        for iy in range(NY):
            g[it, ix, iy] = np.eye(3)

T[:, NX // 2, NY // 2, 0, 0] = 1
g = g + 0.1 * np.random.random(g.shape)


def christoffel(a, b, c):
    # computes the christoffel symbol gammma^a_b_c on the g metric
    global g
    return 0.5 * np.sum(
        [
            g[..., a, d]
            * (
                (
                    np.roll(g[..., d, b], shift=1, axis=c)
                    - np.roll(g[..., d, b], shift=-1, axis=c)
                )
                / D_V[c]
                + (
                    np.roll(g[..., d, c], shift=1, axis=b)
                    - np.roll(g[..., d, c], shift=-1, axis=b)
                )
                / D_V[b]
                - (
                    np.roll(g[..., b, c], shift=1, axis=d)
                    - np.roll(g[..., b, c], shift=-1, axis=d)
                )
                / D_V[d]
            )
            for d in range(3)
        ],
        axis=0,
    )


def update_christoffel():
    global g, G
    for a in range(3):
        for b in range(3):
            for c in range(3):
                G[..., a, b, c] = christoffel(a, b, c)


def update_ricci_curvature():
    global g, G, R
    for a in range(3):
        for b in range(3):
            R[:, :, :, a, b] = np.sum(
                [
                    (
                        np.roll(G[..., c, b, a], shift=1, axis=c)
                        - np.roll(G[..., c, b, a], shift=-1, axis=c)
                    )
                    / D_V[c]
                    - (
                        np.roll(G[..., c, c, a], shift=1, axis=b)
                        - np.roll(G[..., c, c, a], shift=-1, axis=b)
                    )
                    / D_V[b]
                    + np.sum(
                        [
                            G[..., c, c, d] * G[..., d, b, a]
                            - G[..., d, b, c] * G[..., d, c, a]
                            for d in range(3)
                        ],
                        axis=0,
                    )
                    for c in range(3)
                ],
                axis=0,
            )


def scalar_ricci_curvature():
    R_scalar = np.sum(np.sum(g * R, axis=4), axis=3)
    R_scalar_tensor = np.zeros_like(g)
    for a in range(3):
        for b in range(3):
            R_scalar_tensor[..., a, b] = R_scalar

    return R_scalar_tensor


difference = []
for i in tqdm(range(1000)):
    update_christoffel()
    update_ricci_curvature()
    R_scalar = scalar_ricci_curvature()
    g2 = 2 * (R - k * T) / R_scalar
    print(f"max g difference in iteration {i} : {np.max(g2 - g)}")
    difference.append(np.max(g2 - g))
    # g = 0.5 * (g + g2)
    g = 0.9 * g + 0.1 * g2
print("stop here")
plt.plot(difference)
plt.show()


X_MESH, Y_MESH = np.meshgrid(X_VEC, Y_VEC)
u = g[NT // 2, :, :, 0, 0]
v = g[NT // 2, :, :, 1, 0]
plt.quiver(X_MESH, Y_MESH, u, v)
plt.show()
