# Script to generate a numpy array and a .txt file 
# with the coordinates of a carbon nanotube
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse
# Usage: python3 graphene.py -L <L> -m <m> -n <n> 
# C, mass = 12.0107, charge = 0.0
L = 100. # length in A
m = 10 # m chirality index
n = 10 # n chirality index
# chiraliy indices determine archair (m,m), zigzag (m,0) or random (m,n) CNT 
CC = 1.42436 # C-C bond length in A (1.418)

a = CC # unit cell factor

# greatest common divisor 
num1 = 2*m+n
num2 = 2*n+m
while (num1!=num2):
    if (num1>num2): num1 = num1 - num2
    else: num2 = num2 - num1
d_R = num1

# Compute geometric properties
C = a*np.sqrt(3.*(n*n + m*n + m*m))
R = C/(2.*np.pi)
L_cell = np.sqrt(3.)*C/d_R

# fudge radius of the CNT so that bonds are not short
nchord = 2.0*m*np.sqrt(3.*(n*n + m*n + m*m))/np.sqrt(3.*m*m)
rfudge = np.pi/nchord/np.sin(np.pi/nchord)

# Number of unit cells
N_cell = int(np.ceil(L/L_cell))

# index min/max
pmin = 0
pmax = int(np.ceil(n + (n + 2.*m)/d_R))
qmin = int(np.floor(-(2.*n + m)/d_R))
qmax = m

i = 0
coord1 = []
coord2 = []
# generate unit cell coordinates
for q in np.arange(qmin,qmax):
    for p in np.arange(pmin,pmax):
        # first basis atom
        xprime1 = 3.0*a*a*(p*(2.0*n + m) + q*(n + 2.0*m))/(2.0*C)
        yprime1 = 3.0*np.sqrt(3.0)*a*a*(p*m - q*n)/(2.0*C)
        # second basis atom
        xprime2 = xprime1 + 3.0*a*a*(n + m)/(2.0*C)
        yprime2 = yprime1 - a*a*np.sqrt(3.0)*(n - m)/(2.0*C)

        phi1 = xprime1/R
        phi2 = xprime2/R

        if ( (0<=xprime1) and (p*(2.0*n + m) + q*(n + 2.0*m) < 2.0*(n*n + n*m + m*m)) and (0<=yprime1) and (d_R*(p*m-q*n) < 2.0*(n*n+n*m+m*m))):
            coord1.append(np.array([rfudge*R*np.cos(phi1), rfudge*R*np.sin(phi1), yprime1]))
            coord2.append(np.array([rfudge*R*np.cos(phi2), rfudge*R*np.sin(phi2), yprime2]))
            i+=1

Natom = i

# Generate nanotube coordinates
xyzlist = []
for j in range(N_cell):
    for i in range(Natom):
        xyzlist.append(np.array([coord1[i][0], coord1[i][1], coord1[i][2]+j*L_cell]))
        xyzlist.append(np.array([coord2[i][0], coord2[i][1], coord2[i][2]+j*L_cell]))


if (len(xyzlist)!=2*N_cell*Natom): print('Error: Natoms does not match length of xyz list')
Natoms = 2*N_cell*Natom
xyzVec = np.asarray(xyzlist)
#print(xyzVec.shape)
f = open('cnt_{:d}_{:d}_{:d}.xyz'.format(m,n,int(L)),'w')
f.write(str(Natoms)+'\n \n')
for i in range(Natoms):
    f.write('{:s} {:.3f} {:.3f} {:.3f}\n'.format('C',xyzVec[i,0],xyzVec[i,1],xyzVec[i,2]))
f.close()
np.savetxt('cnt_{:d}_{:d}_{:d}.txt'.format(m,n,int(L)), np.c_[xyzVec[:,0], xyzVec[:,1], xyzVec[:,2]], fmt=['%10.3f', '%10.3f', '%10.3f'])

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyzVec[:,0], xyzVec[:,1], xyzVec[:,2])
plt.show()
"""