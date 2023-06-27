import mirheo as mir
import pint
import numpy as np
import h5py
import os
import sys
import shutil
import glob
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if __name__ == "__main__":
    ureg = pint.UnitRegistry()

    # Define Mirheo's unit system.
    ureg.define('mirL = 1 angstrom')
    ureg.define('mirT = 1 fs')
    ureg.define('mirM = 1e-23 g')
    mir.set_unit_registry(ureg)

    path = r'h5_grs_ac/'

    #load graphene particle positions and create new h5 with pulling force
    if rank == 0:
        shutil.rmtree(path, ignore_errors=True) #delete directory before running
        position = np.loadtxt('particle_generators/grs_400_400.txt')
        n = position.shape[0]

        extraforces = np.zeros((n,3), dtype=np.float64)


        
        pulling_force = 1e-4    # 1e-6 N [atm ~1e-3 good]

        armchair = True         #y-direction == Armchair && x-direction == ZigZag
        m = 32                  #amount of pulled particles top & bottom
        
        if armchair:
            unique_sorted_positions = np.sort(np.unique(position[:,1]))
            min_pos = unique_sorted_positions[m]
            max_pos = unique_sorted_positions[-m-1]
            extraforces[position[:,1] < min_pos,:] = np.array([0.0, -pulling_force, 0.0])
            extraforces[position[:,1] > max_pos,:] = np.array([0.0,  pulling_force, 0.0])
        else:
            unique_sorted_positions = np.sort(np.unique(position[:,0]))
            min_pos = unique_sorted_positions[m]
            max_pos = unique_sorted_positions[-m-1]
            extraforces[position[:,0] < min_pos,:] = np.array([-pulling_force, 0.0, 0.0])
            extraforces[position[:,0] > max_pos,:] = np.array([ pulling_force, 0.0, 0.0])

        
        
        shift = np.array([200.0, 200.0, 7.6]) #shift all particles to the center
        for i in range(n):
            position[i,:] += shift

        velocity = np.zeros((n,3), dtype=np.float64)
        id = np.reshape(np.arange(n), (n,1))
        

        os.remove('restart_files/pv.PV-00000.h5')

        f = h5py.File('restart_files/pv.PV-00000.h5', 'a')

        dset = f.create_dataset('position', (n,3), dtype=np.float64)
        dset[...] = position
        dset = f.create_dataset('velocity', (n,3), dtype=np.float64)
        dset[...] = velocity
        dset = f.create_dataset('id', (n,1), dtype=np.int)
        dset[...] = id
        dset = f.create_dataset('extraforces', (n,3), dtype=np.float64)
        dset[...] = extraforces
        
        f.close()
    

    dt = ureg('1 fs')            		   # timestep
    rc = ureg('2.5 angstrom')              # cutoff radius

    ranks  = (1, 1, 1)
    domain = (ureg('800.0 angstrom'), ureg('800.0 angstrom'), ureg('15.0 angstrom'))

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')
    
    pv = mir.ParticleVectors.ParticleVector('pv', mass = ureg('12.0107/6.02214076e23 g'))
    ic = mir.InitialConditions.Restart('restart_files')
    u.registerParticleVector(pv, ic)

    
    #Interactions
    sw2 = mir.Interactions.Pairwise('sw2', rc, kind='SW', epsilon=ureg('1.0 eV'), sigma=ureg('2.025 angstrom'), A=38.39709, B=0.09399)
    u.registerInteraction(sw2)
    u.setInteraction(sw2, pv, pv)

    sw3 = mir.Interactions.Triplewise('sw3', rc, kind='SW3', lambda_=41.0, epsilon=ureg('1.0 eV'), theta=2.0943951023931957, gamma=0.55, sigma=ureg('2.025 angstrom'))
    u.registerInteraction(sw3)
    u.setInteraction(sw3, pv, pv, pv)

    #Integrators
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)


    ####Plugins####
    u.registerPlugins(mir.Plugins.createAddPerParticleForce('extra_force', pv, 'extraforces'))						
    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv], T=ureg('1 K'), tau=ureg('1.0 ps')))

    u.registerPlugins(mir.Plugins.createPlaneOutlet('plane_outlet1', [pv], ( 0.0 , 1.0 , 0.0 ,-790.0)))     #delete the ones that are close to the boundary
    u.registerPlugins(mir.Plugins.createPlaneOutlet('plane_outlet2', [pv], ( 0.0 ,-1.0 , 0.0 ,  10.0)))
    u.registerPlugins(mir.Plugins.createPlaneOutlet('plane_outlet3', [pv], ( 1.0 , 0.0 , 0.0 ,-790.0)))
    u.registerPlugins(mir.Plugins.createPlaneOutlet('plane_outlet4', [pv], (-1.0 , 0.0 , 0.0 ,  10.0)))

    dump_every = 10000
    u.registerPlugins(mir.Plugins.createStats('stats', every=1000))
    u.registerPlugins(mir.Plugins.createForceSaver('forces', pv))
    u.registerPlugins(mir.Plugins.createDumpParticles('part_dump', pv, dump_every, ['forces', 'extraforces'], path + 'pv-'))
    ###############

    u.run(1001, dt=1.0*dt)

    #################################
    #update pulling force
    #################################
    
    for i in range(1000000):
        if u.isComputeTask():
            all_files = glob.glob(path + '*.h5')
            last_file = max(all_files)
            
            g = h5py.File(last_file, 'r')
            
            new_extraforce = np.asarray(g['extraforces'][()])

            print('[max, min]: [', np.max(new_extraforce), ',', np.min(new_extraforce), ']', flush=True)
            
            if new_extraforce.shape[0] < n:
                print('last generated file: ', last_file, flush=True)
                print("graphene sheet broke")
                print("i = ", i)
                break

            pv.additiveUpdateChannel('extraforces', 1e-7)
        u.run(1001, dt=1.0*dt)
    
    sys.exit(0)
    
