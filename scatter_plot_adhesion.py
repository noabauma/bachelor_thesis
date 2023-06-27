from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pint
import numba

ureg = pint.UnitRegistry()

#Hossain 2018 parameters
rc      = 2.5
A       = 38.39709
B       = 0.09399
lambda_ = 41.0
epsilon = 1.0
theta   = 2.0943951023931957
gamma   = 0.55
sigma   = 2.025

#sw 2body force
@numba.njit
def sw2(r_0, r_1): 
    r = np.sqrt(np.dot(r_0,r_1))
    r_vec = r_0 - r_1
    return ( (A*epsilon*sigma*(B*(sigma/r)**4 - 1.0)*np.exp(sigma/(r-rc)))/((r-rc)**2*r) + (A*epsilon*np.exp(sigma/(r-rc))*4.0*B*(sigma/r)**4)/(r*r) )* r_vec

#sw 3body force
@numba.njit
def sw3(r_i, r_j, r_k):
    r_ij_vec = r_i - r_j
    r_jk_vec = r_j - r_k
    r_ki_vec = r_k - r_i

    r_ij = np.sqrt(np.dot(r_ij_vec, r_ij_vec))
    r_jk = np.sqrt(np.dot(r_jk_vec, r_jk_vec))
    r_ki = np.sqrt(np.dot(r_ki_vec, r_ki_vec))

    r_ij_hat = r_ij_vec/r_ij
    r_jk_hat = r_jk_vec/r_jk
    r_ki_hat = r_ki_vec/r_ki

    cos_theta_jik = -np.dot(r_ij_hat, r_ki_hat)
    cos_theta_ijk = -np.dot(r_ij_hat, r_jk_hat)
    cos_theta_ikj = -np.dot(r_ki_hat, r_jk_hat)

    a   = epsilon*lambda_*(cos_theta_jik + 1.0/3.0)**2
    a_  = epsilon*lambda_*(cos_theta_ijk + 1.0/3.0)**2
    a__ = epsilon*lambda_*(cos_theta_ikj + 1.0/3.0)**2

    b   = np.exp(gamma*sigma/(r_ij-rc))
    b_  = np.exp(gamma*sigma/(r_ij-rc))
    b__ = np.exp(gamma*sigma/(r_ki-rc))

    c   = np.exp(gamma*sigma/(r_ki-rc))
    c_  = np.exp(gamma*sigma/(r_jk-rc))
    c__ = np.exp(gamma*sigma/(r_jk-rc))

    THETA   = 2.0*lambda_*epsilon*(cos_theta_jik + 1.0/3.0)*(-r_ki_hat/r_ij + r_ij_hat/r_ki - cos_theta_jik*(r_ij_hat/r_ij - r_ki_hat/r_ki))
    THETA_  = 2.0*lambda_*epsilon*(cos_theta_ijk + 1.0/3.0)*(-r_jk_hat/r_ij - cos_theta_ijk*(r_ij_hat/r_ij))
    THETA__ = 2.0*lambda_*epsilon*(cos_theta_ikj + 1.0/3.0)*( r_jk_hat/r_ki + cos_theta_ikj*(r_ki_hat/r_ki))

    h_jik = b  *c  *(THETA   - a  *(r_ij_hat*gamma*sigma/(r_ij-rc)**2 - r_ki_hat*gamma*sigma/(r_ki-rc)**2))
    h_ijk = b_ *c_ *(THETA_  - a_ *(r_ij_hat*gamma*sigma/(r_ij-rc)**2))
    h_ikj = b__*c__*(THETA__ + a__*(r_ki_hat*gamma*sigma/(r_ki-rc)**2))

    if(r_ij >= rc):
        if(r_jk < rc and r_ki < rc):
            return h_ikj
        else:                           
            return np.zeros((3,))       
    elif(r_jk >= rc and r_ki >= rc):        
        return np.zeros((3,))
    else:                                   
        if(r_jk < rc and r_ki < rc):  
            return h_jik + h_ijk + h_ikj
        elif(r_jk < rc):
            return h_ijk
        else:                              
            return h_jik

#only interact within cutoff radius
@numba.njit
def withinCutOff(r,s):
    rs = r - s
    drs2 = np.dot(rs,rs)
    if drs2 > rc*rc:
        return False
    else:
        return True


#brute force algorithm for calculating interactions
@numba.jit(nopython=True)
def brute_force(dot, particle, centered_particles):
    
    n = dot.shape[0]
    forces = np.zeros((n,3))
    m = particle.shape[0] + centered_particles.shape[0]

    #sw2
    for i in range(n):

        b = centered_particles.shape[0]
        top_layer_particles = np.empty((b,3))
        for l in range(b):
            top_layer_particles[l,:] = centered_particles[l] + dot[i]

        graphene_sheets = np.vstack((top_layer_particles, particle))

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(graphene_sheets[:,0], graphene_sheets[:,1], graphene_sheets[:,2])
        ax.scatter(dot[i,0], dot[i,1], dot[i,2])
        plt.show()
        """
        for j in range(m):
            if withinCutOff(dot[i], graphene_sheets[j]):
                forces[1,:] += sw2(dot[i], graphene_sheets[j])
                #print("%.2f" % (100.0*i/(2*n)), '%', flush=True, end='\r')
    #sw3
    for i in range(n):

        b = centered_particles.shape[0]
        top_layer_particles = np.empty((b,3))
        for l in range(b):
            top_layer_particles[l,:] = centered_particles[l] + dot[i]

        graphene_sheets = np.vstack((top_layer_particles, particle))

        for j in range(m):
            for k in range(j+1,m):
                interact01 = withinCutOff(dot[i], graphene_sheets[j])
                interact12 = withinCutOff(graphene_sheets[j], graphene_sheets[k])
                interact20 = withinCutOff(graphene_sheets[k], dot[i])
                if interact01 and interact12 or interact12 and interact20 or interact20 and interact01:
                    forces[i,:] += sw3(dot[i],graphene_sheets[j],graphene_sheets[k])
                    #print("%.2f" % (100.0*(i+n)/(2*n)), '%', flush=True, end='\r')
                
    return forces

if __name__ == "__main__":
    particles = np.loadtxt('particle_generators/grs_1_20_20.txt')
    plot_particles = particles[particles[:,0]>=2.4,:]
    plot_particles = plot_particles[plot_particles[:,0]<=7.5,:]
    plot_particles = plot_particles[plot_particles[:,1]>=2.4,:]
    plot_particles = plot_particles[plot_particles[:,1]<=7.5,:]

    centered_particles = particles
    for i in range(particles.shape[0]):
        centered_particles[i,:] -= np.array([9.868, 8.546, 0.000])
    centered_particles = np.delete(centered_particles, 88, 0)   #delete (0.0,0.0,0.0)
    
    nx = 40
    ny = 40
    x = np.linspace(2.5, 7.5, nx)
    y = np.linspace(2.5, 7.5, ny)
    z = np.linspace(0.5, 2.5, 9)
    dots = np.zeros((nx*ny*9,3))
    for i in range(nx):
        for j in range(ny):
            for k in range(9):
                dots[i*(ny*9)+j*(9)+k,:] = np.array([x[i], y[j], z[k]])
    
    
    print("calculated forces...")
    forces = brute_force(dots, particles, centered_particles)
 
    forces[1,2] = np.mean(forces[1::9,2][1:])
    #print(forces[1::9,2])
    print("plot figures...")
    fig, axs = plt.subplots(3, 3)
    #fig.suptitle('adhesion with unit [1e-6 N]')

    for i in range(9):
        img = axs[int(i/3), i%3].scatter(x=dots[i::9,0], y=dots[i::9,1], c=forces[i::9,2], label=str(dots[i,2]) + ' $\AA$', vmin=np.min(forces[:,2]), vmax=np.max(forces[:,2]))
        axs[int(i/3), i%3].scatter(x=plot_particles[:,0], y=plot_particles[:,1], c='r', marker='.')
    
    fig.colorbar(img, ax=axs.ravel().tolist())

    for j in range(3):
        for k in range(3):
            axs[j,k].legend(loc='upper right')

    plt.show()
    plt.savefig("scatter_plot_adhesion.png")
    plt.close()





