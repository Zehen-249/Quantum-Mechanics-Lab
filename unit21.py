print("#"*150)
print("\n\nSolution of the 1-D Time Independetnt Schrodinger Equation using Finite Difference Method for Free Particle Problem(Infinite Square Well)\n")
print("#"*150)
print("\n")



# Required Modules
import numpy as np
import matplotlib.pyplot as plt



# Define Constant parameters
h_cut = 1973.0  # (eV*Å)  
m = 0.511e6  # (eV/c^2)

# Define System parameters
x0 = -10.00 
xn = 10.0  
N = 150  # number of points
d = (xn - x0) / (N)  # Step Size
X = np.linspace(x0, xn, N)  

print("\nSystem Parameters and Constants\n")
print(f"h_cut = {h_cut} eV*Å\nMass of particle (m) = {m} eV/c^2\nWidth of the 1-D Square Box from {x0} Å to {xn} Å\n")

# Define Kinetic Energy Matrix
K = np.zeros((len(X), len(X)))  


# Construct the tri-diagonal matrix
for i in range(len(X)):
    for j in range(len(X)):
        if i == j:
            K[i,j] = -2
        elif np.abs(i - j) == 1:
            K[i,j] = 1

# Scale the kinetic energy matrix
K = (-(h_cut**2) / (2 * m * (d**2))) * K

# Construct the Hamiltonian matrix
H = K 

# Solve the eigenvalue equation
eVal, eVec = np.linalg.eig(H)

# get the indices of eigen values in ascending order of energy
z = np.argsort(eVal)

#get the eigen enrgies in ascending order 
energies = eVal[z]

# Print the first three energy levels in eV
print(f"First three energy levels (eV)\n", energies[:3])


#plot the first 4 eigen states 
color=["r",'b','g','b']
plt.figure(figsize=(10,6))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(X, eVec[:,z[i]],label=f"n={i+1}", lw=1,color=color[i])
    plt.axhline(0, color='k' , lw=0.4)
    plt.xlabel("x (Å)")
    plt.ylabel("Wave Function")
    plt.legend()
plt.suptitle("Mehendi Hasan  2230248 \nParticle in a box Equation")

plt.figure(figsize=(10,6))
for i in range(6):
    k=i%4
    plt.axhline(energies[i], color=color[k],label=f"E{i+1}= {energies[i]}",lw=0.4)
    plt.legend()
plt.suptitle("Mehendi Hasan  2230248 \nEnergy levels of the particle in a box")
plt.ylabel("Energy values")
plt.legend()
plt.show()
