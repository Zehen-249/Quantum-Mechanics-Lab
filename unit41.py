import numpy as np

import matplotlib.pyplot as plt

def pot_fn(r,b):
    return ((1/2)*(k*(r**2)) ) + ((1/3)*(b*(r**3)))

# Define Constant parameters
h_cut = 197.3  # MeV*fm
m = 940.0  # MeV/c^2 
k = 100  # MeV/fm^2
b = [0,10,30]   # MeV / fm**3

# Define System parameters
r0 = 1e-3  # fm
rn = 6  # fm
N = 500  # number of points
d = (rn - r0) / (N)  # Step Size
R = np.linspace(r0, rn , N)  # Radial positions

K = np.zeros((3, len(R), len(R)))  # Kinetic energy matrix
V = np.zeros((3, len(R), len(R)))  # Potential energy matrix

for q in range(len(b)):
    for i in range(N):
        for j in range(N):
            if i == j:
                K[q, i, j] = -2
            elif np.abs(i - j) == 1:
                K[q, i, j] = 1

for i in range(len(b)):
    for j in range(len(R)):
        V[i, j, j] = pot_fn(R[j], b[i])

# for k in range(len(b)):
#     for i in range(len(R)):
#         for j in range(len(R)):
#             if i == j:
#                 K[k, i, j] = -2
#                 V[k, i, j] = pot_fn(R[i], b[k])
#             elif np.abs(i - j) == 1:
#                 K[k, i, j] = 1


# Scaling the kinetic energy matrix
K = (-(h_cut**2) / (2 * m * (d**2))) * K

# Constructing the Hamiltonian matrix
H = K + V

# Solve the eigenvalue equation
eVal, eVec = np.linalg.eig(H)

energies = np.zeros_like(eVal)
Z = np.zeros_like(eVal, dtype=int)

# Sorting eigenvalues and corresponding eigenvectors
for k in range(len(b)):
    z = np.argsort(eVal[k])
    Z[k] = z
    energies[k] = eVal[k, z]

print(energies[:,:3])
plt.figure(figsize=(12, 8))

for k in range(len(b)):
    plt.subplot(3, 1, k + 1)
    plt.plot(R, eVec[k, :, Z[k][0]], label=f"Ground state, a = {b[k]} Å", lw=1.5)
    plt.plot(R, eVec[k, :, Z[k][1]], label=f"1st Excited state, a={b[k]} Å", lw=1.5)
    plt.plot(R, eVec[k, :, Z[k][2]], label=f"2nd Excited state, a={b[k]} Å", lw=1.5)
    plt.xlabel("r (Å)")
    plt.ylabel("Wave Function")
    plt.title(f"Wave Functions for Screened Coulomb Potential (b = {b[k]}fm)")
    plt.legend()
    plt.grid(True)

plt.suptitle("Wave Functions from the Radial Schrödinger Equation with Screened Coulomb Potential")
plt.tight_layout(rect=(0., 0., 1., 0.96))
plt.show()

