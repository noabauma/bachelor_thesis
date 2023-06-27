#checking if the analytical gradient is correct


from autograd import grad
import autograd.numpy as np
import sys

np.random.seed(int(sys.argv[1]))


A = np.random.rand()
B = np.random.rand()
epsilon = np.random.rand()
sigma = np.random.rand()
lambda_ = np.random.rand()
gamma = np.random.rand()
rc = 10.0*np.random.rand()

def sw2_pot(r_i, r_j):
    r_ij = r_i - r_j
   
    r = np.sqrt(np.dot(r_ij,r_ij))

    return A*epsilon*(B*((sigma/r)**4) - 1.0)*np.exp(sigma/(r-rc))

def sw3_pot(r_i, r_j, r_k):
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

    h_jik = lambda_*epsilon*(cos_theta_jik + 1.0/3.0)**2 *np.exp(gamma*sigma/(r_ij-rc) + gamma*sigma/(r_ki-rc))
    h_ijk = lambda_*epsilon*(cos_theta_ijk + 1.0/3.0)**2 *np.exp(gamma*sigma/(r_ij-rc) + gamma*sigma/(r_jk-rc))
    h_ikj = lambda_*epsilon*(cos_theta_ikj + 1.0/3.0)**2 *np.exp(gamma*sigma/(r_ki-rc) + gamma*sigma/(r_jk-rc))

    return h_jik + h_ijk + h_ikj


def sw2_force_mirheo(r_i, r_j):
    r_ij = r_i - r_j
    dr_ij2 = np.dot(r_ij, r_ij)
    if dr_ij2 >= rc*rc:
        return np.array([0.0, 0.0, 0.0])
    
    dr_ij = np.sqrt(dr_ij2)

    rs2 = (sigma*sigma) / dr_ij2
    B_rs4 = B * rs2 * rs2
    r_rc = dr_ij - rc
    exp = np.exp(sigma / r_rc)
    A_eps_exp = A * epsilon * exp
    phi = (sigma * (B_rs4 -1.0)*A_eps_exp)/(r_rc*r_rc*dr_ij) + (4.0*B_rs4 * A_eps_exp)/dr_ij2

    return phi * r_ij


def sw3_force_mirheo(r_i, r_j, r_k):
        r_ij = r_i - r_j
        r_jk = r_j - r_k
        r_ki = r_k - r_i
        
        dr_ij2 = np.dot(r_ij, r_ij)
        dr_jk2 = np.dot(r_jk, r_jk)
        dr_ki2 = np.dot(r_ki, r_ki)

        dr_ij = np.sqrt(dr_ij2)
        dr_jk = np.sqrt(dr_jk2)
        dr_ki = np.sqrt(dr_ki2)

        r_ij_hat = r_ij/dr_ij
        r_jk_hat = r_jk/dr_jk
        r_ki_hat = r_ki/dr_ki

        cos_theta_jik = -np.dot(r_ij_hat, r_ki_hat)
        cos_theta_ijk = -np.dot(r_ij_hat, r_jk_hat)
        cos_theta_ikj = -np.dot(r_ki_hat, r_jk_hat)

        dr_ij_inv = 1.0/dr_ij
        dr_jk_inv = 1.0/dr_jk
        dr_ki_inv = 1.0/dr_ki

        dr_ij_rc_inv = 1.0/(dr_ij-rc)
        dr_jk_rc_inv = 1.0/(dr_jk-rc)
        dr_ki_rc_inv = 1.0/(dr_ki-rc)

        #h_jik
        cos_cos = (cos_theta_jik + 1.0/3.0)

        exp = np.exp(gamma*sigma*dr_ij_rc_inv + gamma*sigma*dr_ki_rc_inv)

        exp_lambda_epsilon_cos_cos = exp*lambda_*epsilon*cos_cos

        h_jik_j = exp_lambda_epsilon_cos_cos*(2.0*( r_ki_hat*dr_ij_inv + cos_theta_jik*r_ij_hat*dr_ij_inv) + cos_cos*gamma*sigma*dr_ij_rc_inv*dr_ij_rc_inv*r_ij_hat)
        h_jik_k = exp_lambda_epsilon_cos_cos*(2.0*(-r_ij_hat*dr_ki_inv - cos_theta_jik*r_ki_hat*dr_ki_inv) - cos_cos*gamma*sigma*dr_ki_rc_inv*dr_ki_rc_inv*r_ki_hat)
        h_jik_i = -h_jik_j - h_jik_k

        #h_ijk
        cos_cos = (cos_theta_ijk + 1.0/3.0)

        exp = np.exp(gamma*sigma*dr_ij_rc_inv + gamma*sigma*dr_jk_rc_inv)

        exp_lambda_epsilon_cos_cos = exp*lambda_*epsilon*cos_cos

        h_ijk_i = exp_lambda_epsilon_cos_cos*(2.0*(-r_jk_hat*dr_ij_inv - cos_theta_ijk*r_ij_hat*dr_ij_inv) - cos_cos*gamma*sigma*dr_ij_rc_inv*dr_ij_rc_inv*r_ij_hat)
        h_ijk_k = exp_lambda_epsilon_cos_cos*(2.0*( r_ij_hat*dr_jk_inv + cos_theta_ijk*r_jk_hat*dr_jk_inv) + cos_cos*gamma*sigma*dr_jk_rc_inv*dr_jk_rc_inv*r_jk_hat)
        h_ijk_j = -h_ijk_i - h_ijk_k

        #h_ikj
        cos_cos = (cos_theta_ikj + 1.0/3.0)

        exp = np.exp(gamma*sigma*dr_ki_rc_inv + gamma*sigma*dr_jk_rc_inv)

        exp_lambda_epsilon_cos_cos = exp*lambda_*epsilon*cos_cos

        h_ikj_i = exp_lambda_epsilon_cos_cos*(2.0*( r_jk_hat*dr_ki_inv + cos_theta_ikj*r_ki_hat*dr_ki_inv) + cos_cos*gamma*sigma*dr_ki_rc_inv*dr_ki_rc_inv*r_ki_hat)
        h_ikj_j = exp_lambda_epsilon_cos_cos*(2.0*(-r_ki_hat*dr_jk_inv - cos_theta_ikj*r_jk_hat*dr_jk_inv) - cos_cos*gamma*sigma*dr_jk_rc_inv*dr_jk_rc_inv*r_jk_hat)
        h_ikj_k = -h_ikj_i - h_ikj_j


        if(dr_ij2 >= rc*rc):
            if(dr_jk2 < rc*rc and dr_ki2 < rc*rc):
                return [-h_ikj_i, -h_ikj_j, -h_ikj_k]
            else:                    
                zeros = (0.0, 0.0, 0.0)        
                return [zeros, zeros, zeros]       
            
        elif(dr_jk2 >= rc*rc and dr_ki2 >= rc*rc):
            zeros = (0.0, 0.0, 0.0)        
            return [zeros, zeros, zeros]
        else:                                   
            if(dr_jk2 < rc*rc and dr_ki2 < rc*rc):  
                return [-(h_jik_i + h_ijk_i + h_ikj_i), -(h_jik_j + h_ijk_j + h_ikj_j), -(h_jik_k + h_ijk_k + h_ikj_k)]
            elif(dr_jk2 < rc*rc):
                return [-h_ijk_i, -h_ijk_j, -h_ijk_k]
            else:                              
                return [-h_jik_i, -h_jik_j, -h_jik_k]


def withinCutOff(r,s):
    rs = r - s
    drs2 = np.dot(rs,rs)
    if drs2 > rc*rc :
        return False
    else:
        return True


if __name__ == "__main__":
    n = int(sys.argv[2])

    particle = np.random.rand(n,3)

    grad_sw2_pot = grad(sw2_pot)
    grad_sw3_pot = grad(sw3_pot)


    #autograd algo
    autograd_force = np.zeros((n,3))

    #2body
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if withinCutOff(particle[i], particle[j]):
                autograd_force[i,:] += -grad_sw2_pot(particle[i], particle[j])

    
    #3body
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(j+1, n):
                if k == i:
                    continue
                interact01 = withinCutOff(particle[i], particle[j])
                interact12 = withinCutOff(particle[j], particle[k])
                interact20 = withinCutOff(particle[k], particle[i])
                if interact01 and interact12 or interact12 and interact20 or interact20 and interact01:
                    autograd_force[i,:] += -grad_sw3_pot(particle[i], particle[j], particle[k])
    


    #mirheo algo
    mirheo_force = np.zeros((n,3))

    #2body
    for i in range(n):
        for j in range(i+1, n):
            f_ij = sw2_force_mirheo(particle[i], particle[j])
            mirheo_force[i,:] += f_ij
            mirheo_force[j,:] += -f_ij

    
    #3body
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                interact01 = withinCutOff(particle[i], particle[j])
                interact12 = withinCutOff(particle[j], particle[k])
                interact20 = withinCutOff(particle[k], particle[i])
                if interact01 and interact12 or interact12 and interact20 or interact20 and interact01:
                    f = sw3_force_mirheo(particle[i], particle[j], particle[k])
                    mirheo_force[i,:] += f[0]
                    mirheo_force[j,:] += f[1]
                    mirheo_force[k,:] += f[2]
    


    print("\nautograd   force:\n", autograd_force, "\n")
    print("mirheo force:\n", mirheo_force, "\n")
    #print("differences:\n", np.abs(autograd_force - mirheo_force))

    