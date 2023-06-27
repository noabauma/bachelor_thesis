import matplotlib.pyplot as plt
import numpy as np
import pint
import h5py
import glob

if __name__ == "__main__":
    ureg = pint.UnitRegistry()

    for path in [r'h5_grs_ac/', r'h5_grs_zz/']:
        all_files = glob.glob(path + "*.h5")
        
        n_files = len(all_files)

        all_files = np.sort(all_files)

        stress = np.empty(0)
        strain = np.empty(0)

        first = True
        for i in range(n_files-1):
            f = h5py.File(all_files[i], "r")
            f_ = h5py.File(all_files[i+1], "r")

            #initial file
            if i == 0:
                armchair = True if np.max(f_['extraforces'][()][:, 1]) != 0.0 else False
                print("Armchair" if armchair else "ZigZag")
                L_0 = (np.max(f["position"][()][:,int(armchair)])     - np.min(f["position"][()][:,int(armchair)]))*ureg('1.0 angstrom')
                W_0 = (np.max(f["position"][()][:,int(not armchair)]) - np.min(f["position"][()][:,int(not armchair)]))*ureg('1.0 angstrom')
                H_0 = ureg('3.4 angstrom')   # 2 * Van der Waals radii of carbon (like in the paper)

                A_0 = W_0*H_0   #cross section

                print("equilibrium area: ", A_0)
                print("how many particles where pulled in one direction: ", f['extraforces'][()][f['extraforces'][()][:,int(armchair)]>0.0, int(armchair)].shape[0])
            

            if(np.max(f['extraforces'][()]) != np.max(f_['extraforces'][()])):

                L = (np.max(f["position"][()][:,int(armchair)]) - np.min(f["position"][()][:,int(armchair)]))*ureg('1.0 angstrom')

                total_pulling_force = np.sum(f['extraforces'][()][f['extraforces'][()][:,int(armchair)]>0.0, int(armchair)])*ureg('1e-06 N')
                
                
                if first:
                    stress = np.append(stress, 0.0)
                    first = False
                else:
                    stress = np.append(stress, total_pulling_force/(A_0))
                
                strain = np.append(strain, (L-L_0)/L_0)
        #plot
        plt.plot(strain, stress.to(ureg.GPa), color='g' if armchair else 'r', label="Mirheo Armchair" if armchair else "Mirheo ZigZag")

    #paper comparison plots
    hossain_ac = np.loadtxt('hossain/Hossain_stress_strain_ac.csv', delimiter=',')
    hossain_zz = np.loadtxt('hossain/Hossain_stress_strain_zz.csv', delimiter=',')
    plt.plot(hossain_ac[:,0], hossain_ac[:,1], 'gx', label='Hossain et al. (2018), Armchair')
    plt.plot(hossain_zz[:,0], hossain_zz[:,1], 'rx', label='Hossain et al. (2018), ZigZag')
    
    plt.xlabel('Strain')
    plt.ylabel('Stress [GPa]')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig("stress-strain.png")
    plt.show()
        
#Armchair strongest pulled was 0.0003227699999998847*1E-6 N per particle (5216 Particles) -> ~123.5 GPa