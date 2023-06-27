import matplotlib.pyplot as plt
import numpy as np
import h5py
import glob
import numba
import math


#all the parameters
A = 38.39709
B = 0.09399
epsilon = 1.60217663e-19       #joule (1.0 eV)
sigma = 2.025                  #angstrom
lambda_ = 41.0
gamma = 0.55
rc = 2.5                       #angstrom
rc2 = rc*rc                    #angstrom**2


sigma_lj = 3.3611                     #angstrom
eps = 0.097                           #kcal/mol
epsilon_lj = 4184*eps/6.02214076e23   #joule

@numba.njit
def lj_pot(r):
    sr6 = (sigma_lj/r)**6
    pot = 4.*epsilon_lj*(sr6*sr6 - sr6)
    return pot


@numba.njit
def l2_distance(atom1, atom2):
    dx = atom2[0] - atom1[0]
    dy = atom2[1] - atom1[1]
    dz = atom2[2] - atom1[2]

    r = (dx * dx + dy * dy + dz * dz) ** 0.5

    return r

@numba.njit
def l2_nonsqrt_distance(atom1, atom2):
    dx = atom2[0] - atom1[0]
    dy = atom2[1] - atom1[1]
    dz = atom2[2] - atom1[2]

    r = (dx * dx + dy * dy + dz * dz)

    return r


@numba.njit(parallel=True)
def tot_pot(cluster):
    energy = 0.0
    # numba.prange requires parallel=True flag to compile.
    # It causes the loop to run in parallel in multiple threads.
    for i in numba.prange(len(cluster)-1):
        for j in range(i + 1, len(cluster)):
            r = l2_distance(cluster[i], cluster[j])
            e = lj_pot(r)
            energy += e

    return energy


@numba.njit(parallel=True)
def tot_pot_2(cluster_1, cluster_2):
    energy = 0.0
    # numba.prange requires parallel=True flag to compile.
    # It causes the loop to run in parallel in multiple threads.
    for i in numba.prange(len(cluster_1)):
        for j in range(len(cluster_2)):
            r = l2_distance(cluster_1[i], cluster_2[j])
            if r >= 12.0:
                continue
            energy += lj_pot(r)

    return energy


@numba.njit
def sw2_pot(r):
    return A*epsilon*(B*((sigma/r)**4) - 1.0)*np.exp(sigma/(r-rc))


@numba.njit
def all_h(r_i, r_j, r_k):
    r_ij_vec = r_i - r_j
    r_jk_vec = r_j - r_k
    r_ki_vec = r_k - r_i

    r_ij = l2_distance(r_i, r_j)
    r_jk = l2_distance(r_j, r_k)
    r_ki = l2_distance(r_k, r_i)

    r_ij_hat = r_ij_vec/r_ij
    r_jk_hat = r_jk_vec/r_jk
    r_ki_hat = r_ki_vec/r_ki

    cos_theta_jik = -np.dot(r_ij_hat, r_ki_hat)
    cos_theta_ijk = -np.dot(r_ij_hat, r_jk_hat)
    cos_theta_ikj = -np.dot(r_ki_hat, r_jk_hat)

    h_jik = lambda_*epsilon*(cos_theta_jik + 1.0/3.0)**2 *np.exp(gamma*sigma/(r_ij-rc) + gamma*sigma/(r_ki-rc))
    h_ijk = lambda_*epsilon*(cos_theta_ijk + 1.0/3.0)**2 *np.exp(gamma*sigma/(r_ij-rc) + gamma*sigma/(r_jk-rc))
    h_ikj = lambda_*epsilon*(cos_theta_ikj + 1.0/3.0)**2 *np.exp(gamma*sigma/(r_ki-rc) + gamma*sigma/(r_jk-rc))

    return h_jik + h_ijk + h_ikj

@numba.njit
def h_jik(r_i, r_j, r_k):
    r_ij_vec = r_i - r_j
    r_ki_vec = r_k - r_i

    r_ij = l2_distance(r_i, r_j)
    r_ki = l2_distance(r_k, r_i)

    r_ij_hat = r_ij_vec/r_ij
    r_ki_hat = r_ki_vec/r_ki

    cos_theta_jik = -np.dot(r_ij_hat, r_ki_hat)

    h_jik = lambda_*epsilon*(cos_theta_jik + 1.0/3.0)**2 *np.exp(gamma*sigma/(r_ij-rc) + gamma*sigma/(r_ki-rc))
    
    return h_jik

@numba.njit
def h_ijk(r_i, r_j, r_k):
    r_ij_vec = r_i - r_j
    r_jk_vec = r_j - r_k

    r_ij = l2_distance(r_i, r_j)
    r_jk = l2_distance(r_j, r_k)

    r_ij_hat = r_ij_vec/r_ij
    r_jk_hat = r_jk_vec/r_jk

    cos_theta_ijk = -np.dot(r_ij_hat, r_jk_hat)

    h_ijk = lambda_*epsilon*(cos_theta_ijk + 1.0/3.0)**2 *np.exp(gamma*sigma/(r_ij-rc) + gamma*sigma/(r_jk-rc))

    return h_ijk

@numba.njit
def h_ikj(r_i, r_j, r_k):
    r_jk_vec = r_j - r_k
    r_ki_vec = r_k - r_i

    r_jk = l2_distance(r_j, r_k)
    r_ki = l2_distance(r_k, r_i)

    r_jk_hat = r_jk_vec/r_jk
    r_ki_hat = r_ki_vec/r_ki

    cos_theta_ikj = -np.dot(r_ki_hat, r_jk_hat)

    h_ikj = lambda_*epsilon*(cos_theta_ikj + 1.0/3.0)**2 *np.exp(gamma*sigma/(r_ki-rc) + gamma*sigma/(r_jk-rc))
    
    return h_ikj


@numba.njit(parallel=True)
def brute_force_pot(position):
    n = len(position)
    energy = 0.0
    #2Body
    for i in numba.prange(n-1):
        for j in range(i+1, n):
            r2 = l2_nonsqrt_distance(position[i], position[j])
            if r2 >= rc2:
                continue
            r = math.sqrt(r2)
            energy += sw2_pot(r)

    #3Body
    for i in numba.prange(n-1):
        for j in range(i+1, n):
            for k in range(j+1,n):
                r_ij2 = l2_nonsqrt_distance(position[i], position[j])
                r_jk2 = l2_nonsqrt_distance(position[j], position[k])
                r_ki2 = l2_nonsqrt_distance(position[k], position[i])
                if r_ij2 < rc2 and r_jk2 < rc2 and r_ki2 < rc2:
                    energy += all_h(position[i], position[j], position[k])
                elif r_ij2 < rc2 and r_jk2 < rc2:
                    energy += h_ijk(position[i], position[j], position[k])
                elif r_jk2 < rc2 and r_ki2 < rc2:
                    energy += h_ikj(position[i], position[j], position[k])
                elif r_ki2 < rc2 and r_ij2 < rc2:
                    energy += h_jik(position[i], position[j], position[k])
        


    return energy
                


def eps(file_name):
    n = len(file_name)
    count = 0
    idx_beg = 0
    idx_end = 0
    for i in range(n):
        if file_name[i] == '/' and count == 0:
            idx_beg = i+1
            count += 1
        elif file_name[i] == '/' and count == 1:
            idx_end = i
            break
    return file_name[idx_beg:idx_end]


if __name__ == "__main__":
    path = r'h5_adhesion/'

    all_files = glob.glob(path + '/*/*.h5')
    all_files = np.sort(all_files)
    n = len(all_files)
    
    tot_pot_energy = []
    epsilons = []
    for i in range(n-1):
        eps_1 = eps(all_files[i])
        eps_2 = eps(all_files[i+1])

        if(eps_1 != eps_2):

            g = h5py.File(all_files[i-6], 'r')
            position_1 = np.asarray(g['position'][()])
            g = h5py.File(all_files[i], 'r')
            position_2 = np.asarray(g['position'][()])

            energy = tot_pot_2(position_1, position_2)
            
            U_e = -1000*energy/(50*50*1e-20)

            tot_pot_energy = np.append(tot_pot_energy, U_e)
            epsilons = np.append(epsilons, float(eps_1))

            #print("Tot. pot. energy: ", tot_pot_energy)
            #print("adhesion [mJ/m^2]: ", U_e)
    
    #tot_pot_energy = np.reshape(tot_pot_energy, (tot_pot_energy.shape[0], 1))
    
    m, c = np.linalg.lstsq(np.vstack([epsilons, np.ones(epsilons.shape[0]).T]).T, tot_pot_energy, rcond=None)[0]

    plt.plot(epsilons, epsilons*m + c, 'b--')

    #only use five reference points
    bag = 10
    jump = int(epsilons.shape[0]/(bag))
    epsilons_avg = np.empty(bag)
    tot_pot_energy_avg = np.empty(bag)
    for i in range(bag):
        upto = min((i+1)*jump , epsilons.shape[0])
        epsilons_avg[i] = np.mean(epsilons[i*jump:upto])
        tot_pot_energy_avg[i] = np.mean(tot_pot_energy[i*jump:upto])
    
    #plt.plot(epsilons, tot_pot_energy, 'r-o', label='original')
    plt.plot(epsilons_avg, tot_pot_energy_avg, 'go')
    plt.xlabel('$\epsilon \: [kcal/mol]$')
    plt.ylabel('$U_e \: [mJ/m^2]$')
    #plt.title('Adhesion Energy of two graphene sheet (size (50A)**2)')
    plt.grid()
    #plt.legend(loc='best')
    plt.ylim(385.0, 395.0)
    plt.savefig('adhesion.png')
    plt.show()
    plt.close()
            


#unused methods

#print("SW tot pot for one sheet", brute_force_pot(position_1))
#position = np.vstack((position_1, position_2))
#tot_pot_energy = tot_pot(position)
        