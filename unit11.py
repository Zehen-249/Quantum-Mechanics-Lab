import numpy as np
import matplotlib.pyplot as plt
# from scipy.special import sph_harm


def fact(x):
    if x==1 or x==0:
        return 1
    else:
        return x*fact(x-1)


def find_factors(n):
   # Initialize the best pair of factors
    best_a, best_b = 1, n  # Starting with the trivial factor pair 1 and n
    
    # Iterate over possible divisors up to sqrt(n)
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:  # If i is a divisor
            a = i
            b = n // i  # The corresponding factor
            # Check if this pair is closer to being equal than the previous best
            if abs(a - b) < abs(best_a - best_b):
                best_a, best_b = a, b
    
    return best_a, best_b
    

def sph_harm(l,m,theta,phi):
    N = np.sqrt((2*l + fact(l-np.abs(m)))/(4*np.pi + fact(l+np.abs(m))))

    # return N * asso_legendre_pol(l,m,np.cos(theta)) * np.exp(1j*m*phi)
    return np.real(N * asso_legendre_pol(l,m,np.cos(theta)) * np.exp(1j*m*phi))

def legendre_poly(l,x):
    if l == 0:
        return 1
    elif l == 1:
        return x
    else:
        return ((2 * l - 1) * x * legendre_poly(l - 1, x) - (l - 1) * legendre_poly(l - 2, x)) / l

def asso_legendre_pol(l,m,x):
    P_lx = legendre_poly(l, x)
    
    result = P_lx
    
    for i in range(np.abs(m)):
        delta = 1e-6
        # finite difference to approximate derivatives manually
        result = (legendre_poly(l, x + delta) - legendre_poly(l, x - delta)) / (2 * delta)
    
    return ((1 - x**2)**(np.abs(m) / 2)) * result



def plot_sphr_harm(l):
    # Generate theta and phi angles
    theta = np.linspace(0, np.pi, 50)     # polar angle
    phi = np.linspace(0, 2 * np.pi, 50)   # azimuthal angle
    theta, phi = np.meshgrid(theta, phi)

    for i in range(len(l)):
        m = []  # allowed magnetic quantum numbers
        for j in range(-l[i],l[i]+1):
            m.append(j)
    
        # Plot using Matplotlib's 3D plotting
       
        row,cols = find_factors(2*l[i] + 1)

        # iterate all values of m and plot spherical harmonics
        for j in range(len(m)):
            # Calculate the spherical harmonic Y(l, m)
            Y_lm = sph_harm(l[i], m[j], theta, phi)
            # Y_lm = sph_harm(m[j],l[i], phi,theta)

            # Calculate the magnitude (for the radial distance)
            r = np.abs(Y_lm)

            # Convert spherical to Cartesian coordinates for plotting
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1 ,1, 1, projection='3d')
            ax.plot_surface(x, y, z, rstride=1, cstride=1, color='black', edgecolor='white', alpha=0.6)

            # Adjust plot view for better visualization
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Y({l[i]},{m[j]})")

        plt.show()

        
    

l = [3,2]  # angular quantum number
plot_sphr_harm(l)
