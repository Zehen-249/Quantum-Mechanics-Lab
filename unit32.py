import numpy as np
import matplotlib.pyplot as plt

def pot_fn(r, a):
    if r == 0:
        return 0
    return (-(e**2) / r) * np.exp(-r / a)

# Define Constant parameters
e = 3.795  # in eV*Å**(1/2)
h_cut = 1973.0  # (eV*A)
m = 0.511e6  # (eV/c^2)
a = [3,5,7]  # in Å

# Define System parameters
r0 = 0.00
rn = 10.0
N = 300  # number of points
d = (rn - r0) / (N)  # Step Size
R = np.linspace(r0 + d, rn - d, N - 2)  # Radial positions

K = np.zeros((3, len(R), len(R)))  # Kinetic energy matrix
V = np.zeros((3, len(R), len(R)))  # Potential energy matrix

for k in range(len(a)):
    for i in range(len(R)):
        for j in range(len(R)):
            if i == j:
                K[k, i, j] = -2
                V[k, i, j] = pot_fn(R[i], a[k])
            elif np.abs(i - j) == 1:
                K[k, i, j] = 1

# Scaling the kinetic energy matrix
K = (-(h_cut**2) / (2 * m * (d**2))) * K

# Constructing the Hamiltonian matrix
H = K + V

# Solve the eigenvalue equation
eVal, eVec = np.linalg.eig(H)

energies = np.zeros_like(eVal)
Z = np.zeros_like(eVal, dtype=int)

# Sorting eigenvalues and corresponding eigenvectors
for k in range(len(a)):
    z = np.argsort(eVal[k])
    Z[k] = z
    energies[k] = eVal[k, z]

print(energies[:,:3])
plt.figure(figsize=(12, 8))

for k in range(len(a)):
    plt.subplot(3, 1, k + 1)
    plt.plot(R, eVec[k, :, Z[k][0]], label=f"Ground state, a={a[k]} Å", lw=1.5)
    plt.plot(R, eVec[k, :, Z[k][1]], label=f"1st Excited state, a={a[k]} Å", lw=1.5)
    plt.plot(R, eVec[k, :, Z[k][2]], label=f"2nd Excited state, a={a[k]} Å", lw=1.5)
    plt.xlabel("r (Å)")
    plt.ylabel("Wave Function")
    plt.title(f"Wave Functions for Screened Coulomb Potential (a = {a[k]} Å)")
    plt.legend()
    plt.grid(True)

plt.suptitle("Wave Functions from the Radial Schrödinger Equation with Screened Coulomb Potential")
plt.tight_layout(rect=(0., 0., 1., 0.96))
plt.show()
