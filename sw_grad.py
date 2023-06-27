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
rc = np.random.rand()

def sw_2body_pot(r_vec):
    r = np.sqrt(np.dot(r_vec,r_vec))
    return A*epsilon*(B*((sigma/r)**4) - 1.0)*np.exp(sigma/(r-rc))


def sw_2body_force(r_vec): 
    r = np.sqrt(np.dot(r_vec,r_vec))
    return ( (A*epsilon*sigma*(B*(sigma/r)**4 - 1.0)*np.exp(sigma/(r-rc)))/((r-rc)**2*r) + (A*epsilon*np.exp(sigma/(r-rc))*4.0*B*(sigma/r)**4)/(r*r) )* r_vec
    #return ( A*epsilon*B*(sigma/r)**4*np.exp(sigma/(r-rc))*(4/r + sigma/(r-rc)**2) - A*epsilon*sigma/(r-rc)**2*np.exp(sigma/(r-rc)) ) * r_vec/r




def sw_3body_pot(r_i, r_j, r_k):
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

def sw_3body_force(r_i, r_j, r_k):
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

    h_jik_r = b  *c  *(THETA   - a  *(r_ij_hat*gamma*sigma/(r_ij-rc)**2 - r_ki_hat*gamma*sigma/(r_ki-rc)**2))
    h_ijk_r = b_ *c_ *(THETA_  - a_ *(r_ij_hat*gamma*sigma/(r_ij-rc)**2))
    h_ikj_r = b__*c__*(THETA__ + a__*(r_ki_hat*gamma*sigma/(r_ki-rc)**2))

    return -(h_jik_r + h_ijk_r + h_ikj_r)

def sw3_jki(r_j, r_k, r_i):
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

    THETA   = 2.0*lambda_*epsilon*(cos_theta_jik + 1.0/3.0)*( r_ki_hat/r_ij + cos_theta_jik*(r_ij_hat/r_ij))
    THETA_  = 2.0*lambda_*epsilon*(cos_theta_ijk + 1.0/3.0)*( r_jk_hat/r_ij - r_ij_hat/r_jk - cos_theta_ijk*(r_jk_hat/r_jk - r_ij_hat/r_ij))
    THETA__ = 2.0*lambda_*epsilon*(cos_theta_ikj + 1.0/3.0)*(-r_ki_hat/r_jk - cos_theta_ikj*(r_jk_hat/r_jk))

    h_jik_r = b  *c  *(THETA   + a  *(r_ij_hat*gamma*sigma/(r_ij-rc)**2))
    h_ijk_r = b_ *c_ *(THETA_  + a_ *(r_ij_hat*gamma*sigma/(r_ij-rc)**2 - r_jk_hat*gamma*sigma/(r_jk-rc)**2))
    h_ikj_r = b__*c__*(THETA__ - a__*(r_jk_hat*gamma*sigma/(r_jk-rc)**2))
    
    return -(h_jik_r + h_ijk_r + h_ikj_r)

def cpp(r_i, r_j, r_k):
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

        return [-(h_jik_i + h_ijk_i + h_ikj_i), -(h_jik_j + h_ijk_j + h_ikj_j), -(h_jik_k + h_ijk_k + h_ikj_k)]

if __name__ == "__main__":
    r = np.random.rand(3)
    grad_sw_2body_pot = grad(sw_2body_pot)
    print("sw 2body autograd   force: ", -grad_sw_2body_pot(r))
    print("sw 2body analytical force: ", sw_2body_force(r))

    r_i = np.random.rand(3)
    r_j = np.random.rand(3)
    r_k = np.random.rand(3)

    print("3Body SW...")
    grad_sw_3body_pot = grad(sw_3body_pot)
    #print("sw 3body autograd   force: ", -grad_sw_3body_pot(r_i, r_j, r_k))
    #print("sw 3body analytical force: ", sw_3body_force(r_i, r_j, r_k))

    print("newton's 3rd law: ", -grad_sw_3body_pot(r_i, r_j, r_k) - grad_sw_3body_pot(r_j, r_k, r_i) - grad_sw_3body_pot(r_k, r_i, r_j))
    print("newton's 3rd law: ",  sw_3body_force(r_i, r_j, r_k)    + sw_3body_force(r_j, r_k, r_i)    + sw_3body_force(r_k, r_i, r_j))
    
    """
    print("(y,n,y)")
    print("h_jik with kernel(i,j,k): ", h_jik_r(r_i, r_j, r_k))
    print("h_ikj with kernel(j,k,i): ", h_ikj_r(r_j, r_k, r_i))
    print("h_ijk with kernel(k,i,j): ", h_ijk_r(r_k, r_i, r_j))
    print(-h_jik_r(r_i, r_j, r_k) - h_ikj_r(r_j, r_k, r_i))

    print("(n,y,y)")
    print("h_ikj with kernel(i,j,k): ", h_ikj_r(r_i, r_j, r_k))
    print("h_ijk with kernel(j,k,i): ", h_ijk_r(r_j, r_k, r_i))
    print("h_jik with kernel(k,i,j): ", h_jik_r(r_k, r_i, r_j))
    print(-h_ikj_r(r_i, r_j, r_k) - h_ijk_r(r_j, r_k, r_i))

    print("(y,y,n)")
    print("h_ijk with kernel(i,j,k): ", h_ijk_r(r_i, r_j, r_k))
    print("h_jik with kernel(j,k,i): ", h_jik_r(r_j, r_k, r_i))
    print("h_ikj with kernel(k,i,j): ", h_ikj_r(r_k, r_i, r_j))
    print(-h_ijk_r(r_i, r_j, r_k) - h_jik_r(r_j, r_k, r_i))
    """
    print("kernel(i,j,k) as kernel(j,k,i)", sw_3body_force(r_j, r_k, r_i))
    #print(h_ikj_r(r_i, r_j, r_k))
    print("kernel(j,k,i): ", sw3_jki(r_j, r_k, r_i))

    print("newton's 3rd law: ", -grad_sw_3body_pot(r_i, r_j, r_k), - grad_sw_3body_pot(r_j, r_k, r_i), - grad_sw_3body_pot(r_k, r_i, r_j))
    print("cpp implementation: ", cpp(r_i, r_j, r_k))
    print("cpp implementation: ", cpp(r_i, r_k, r_j))