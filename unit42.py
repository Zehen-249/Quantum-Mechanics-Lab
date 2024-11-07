import numpy as np
import matplotlib.pyplot as plt

def r_dash(r):
    if r == 0 : 
      return "division by zero error"
    return (r-r_)/r

def pot_fn(r):
    return D*(np.exp(-2*a*r_dash(r)) - np.exp(-a*r_dash(r)))

# Define Constant parameters
h_cut = 1973.0  # (eV*A)  
D = 0.755501
m = 940e6  # (eV/c^2)
a=1.44
r_ = 0.131349

# Define System parameters
r0 = 1.00e-7
rn = 10 
N = 400  # number of points
d = (rn - r0) / (N)  # Step Size
R = np.linspace(r0, rn, N)  # Radial positions

K = np.zeros((len(R), len(R)))  # Kinetic energy matrix
V = np.zeros((len(R), len(R)))  # Potential energy matrix

for i in range(len(R)):
    for j in range(len(R)):
        if i == j:
            K[i,j] = -2
            V[i,j] = pot_fn(r_dash(R[i]))
        elif np.abs(i - j) == 1:
            K[i,j] = 1

# Scaling the kinetic energy matrix
K = (-(h_cut**2) / (2 * m * (d**2))) * K

# Constructing the Hamiltonian matrix
H = K + V

# Solve the eigenvalue equation
eVal, eVec = np.linalg.eig(H)
z = np.argsort(eVal)
energies = eVal[z]

# Print the first three energy levels in eV
print("First three energy levels (eV):", energies[:3])

#plot the first 4 eigen states 
plt.figure(figsize=(10,6))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(R, eVec[:,z[i]],label=f"n={i+1}, r={R[i]}", lw=1)
    plt.xlabel("r (Å)")
    plt.ylabel("Wave Function")
    plt.legend()
plt.suptitle("s-Wave Schrodinger's Equation")

plt.figure(figsize=(10,6))
for i in range(10):
    plt.axhline(energies[i], color='r')
plt.ylabel("Energy values")
plt.legend()
plt.show()