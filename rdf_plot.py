import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

path = r'rdf_2/'
all_files = glob.glob(path + "*.csv")

nbins = 1000
g_r = np.zeros(nbins)
r   = np.empty(nbins)
r_bool = True

for filename in all_files:
    df = pd.read_csv(filename, index_col=None)
    temp = df.to_numpy()
    if(r_bool):
        r_bool = False
        r = temp[:,0]
    #print(temp.shape)
    g_r += temp[:,1]

g_r /= len(all_files)

#include plots from the paper
x = np.loadtxt('mW/rdf_mW.csv', delimiter=',')
plt.plot(x[:,0],x[:,1], 'g.', label="RDF from Molinero et al. (2009)")
#include plots from the paper
x = np.loadtxt('mW/rdf_X-Ray.csv', delimiter=',')
plt.plot(x[:,0],x[:,1], 'yo', label="X-Ray from Narten et al. (1971)")
#plot Mirheo rdf at last
plt.plot(10.0*r, g_r, 'r', label="Mirheo")

plt.xlabel('r [Angstrom]')
plt.ylabel('g(r)')
#plt.title('Radial Distribution Function (rdf) for SW-Potential')
plt.xlim((2,9))
plt.legend(loc='best')
plt.grid()
plt.savefig('rdf_mW_2.png')
#plt.show()
plt.close()
