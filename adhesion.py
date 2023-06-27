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
    eps = float(sys.argv[1])/1000.0
    subpath = str(eps) + '/'

    ureg = pint.UnitRegistry()

    # Define Mirheo's unit system.
    ureg.define('mirL = 1 angstrom')
    ureg.define('mirT = 1 fs')
    ureg.define('mirM = 1e-23 g')
    mir.set_unit_registry(ureg)

    path = r'h5_adhesion/' + subpath

    #load graphene particle positions and create new h5 with pulling force
    if rank == 0:
        shutil.rmtree(path, ignore_errors=True) #delete directory before running

        position = np.loadtxt('particle_generators/grs_2_50_50.txt')

        #bottom sheet
        position_1 = position[position[:,2] < 0.1,:]
        n_1 = position_1.shape[0]

        #top sheet
        position_2 = position[position[:,2] > 0.1,:]
        n_2 = position_2.shape[0]

        if n_1 == n_2:
            n = n_1
        else:
            sys.exit('error, both sheets don\'t have the same number of particles!',0)


        extraforces = np.zeros((n,3), dtype=np.float64)


        """
        extra_force = 0   # 1e-6 N (maybe needed)
        extraforces[position[:,2] < 0.1,:] = np.array([0.0, 0.0,  extra_force])
        extraforces[position[:,2] > 0.1,:] = np.array([0.0, 0.0, -extra_force])
        """
        
        shift = np.array([200.0, 200.0, 48.325]) #shift all particles to the center
        for i in range(n):
            position_1[i,:] += shift
            position_2[i,:] += shift

        velocity = np.zeros((n,3), dtype=np.float64)
        id = np.reshape(np.arange(n), (n,1))
        
        #bottom sheet
        os.remove('restart_files_adhesion_1/pv1.PV-00000.h5')
        
        f = h5py.File('restart_files_adhesion_1/pv1.PV-00000.h5', 'a')

        dset = f.create_dataset('position', (n,3), dtype=np.float64)
        dset[...] = position_1
        dset = f.create_dataset('velocity', (n,3), dtype=np.float64)
        dset[...] = velocity
        dset = f.create_dataset('id', (n,1), dtype=np.int)
        dset[...] = id
        dset = f.create_dataset('extraforces', (n,3), dtype=np.float64)
        dset[...] = extraforces
        
        f.close()

        #top sheet
        os.remove('restart_files_adhesion_2/pv2.PV-00000.h5')

        f = h5py.File('restart_files_adhesion_2/pv2.PV-00000.h5', 'a')

        dset = f.create_dataset('position', (n,3), dtype=np.float64)
        dset[...] = position_2
        dset = f.create_dataset('velocity', (n,3), dtype=np.float64)
        dset[...] = velocity
        dset = f.create_dataset('id', (n,1), dtype=np.int)
        dset[...] = id
        dset = f.create_dataset('extraforces', (n,3), dtype=np.float64)
        dset[...] = extraforces
        
        f.close()
    
    
    


    dt = ureg('1 fs')            		    # timestep
    rc = ureg('2.5 angstrom')              # cutoff radius

    ranks  = (1, 1, 1)
    domain = (ureg('800.0 angstrom'), ureg('800.0 angstrom'), ureg('100.0 angstrom'))

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')
    
    pv1 = mir.ParticleVectors.ParticleVector('pv1', mass = ureg('12.0107/6.02214076e23 g')) #lower sheet
    pv2 = mir.ParticleVectors.ParticleVector('pv2', mass = ureg('12.0107/6.02214076e23 g')) #upper sheet
    ic1 = mir.InitialConditions.Restart('restart_files_adhesion_1')
    ic2 = mir.InitialConditions.Restart('restart_files_adhesion_2')
    u.registerParticleVector(pv1, ic1)
    u.registerParticleVector(pv2, ic2)

    
    #Interactions
    sw2 = mir.Interactions.Pairwise('sw2', rc, kind='SW', epsilon=ureg('1.0 eV'), sigma=ureg('2.025 angstrom'), A=38.39709, B=0.09399)
    u.registerInteraction(sw2)
    u.setInteraction(sw2, pv1, pv1)
    u.setInteraction(sw2, pv2, pv2)

    sw3 = mir.Interactions.Triplewise('sw3', rc, kind='SW3', lambda_=41.0, epsilon=ureg('1.0 eV'), theta=2.0943951023931957, gamma=0.55, sigma=ureg('2.025 angstrom'))
    u.registerInteraction(sw3)
    u.setInteraction(sw3, pv1, pv1, pv1)
    u.setInteraction(sw3, pv2, pv2, pv2)



    #lj  = mir.Interactions.Pairwise('lj', rc=ureg('12.0 angstrom'), kind='LJ', epsilon=ureg('0.82/6.02214076e23 kcal'), sigma=ureg('3.46 angstrom'))   #from paper
    #lj  = mir.Interactions.Pairwise('lj', rc=ureg('12.0 angstrom'), kind='LJ', epsilon=eps*ureg('1.0/6.02214076e23 kcal'), sigma=ureg('3.3611 A'))
    lj  = mir.Interactions.Pairwise('lj', rc=ureg('12.0 angstrom'), kind='LJ', epsilon=eps*ureg('1.0/6.02214076e23 kcal'), sigma=ureg('3.4683 angstrom'))
    u.registerInteraction(lj)
    u.setInteraction(lj, pv1, pv2)

    #Integrators
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv1)
    u.setIntegrator(vv, pv2)


    ####Plugins####
    u.registerPlugins(mir.Plugins.createAddPerParticleForce('extra_force', pv1, 'extraforces'))						
    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv1, pv2], T=ureg('10.0 K'), tau=ureg('1.0 ps')))
    
    u.registerPlugins(mir.Plugins.createPlaneOutlet('plane_outlet1', [pv1, pv2], ( 0.0 , 0.0 ,  1.0 , -99.0))) #delete the ones that are close to the boundary
    u.registerPlugins(mir.Plugins.createPlaneOutlet('plane_outlet2', [pv1, pv2], ( 0.0 , 0.0 , -1.0 ,  1.0)))

    dump_every = 2000
    u.registerPlugins(mir.Plugins.createStats('stats', every=2000))
    u.registerPlugins(mir.Plugins.createForceSaver('forces_1', pv1))
    u.registerPlugins(mir.Plugins.createForceSaver('forces_2', pv2))
    u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_1', pv1, dump_every, ['forces', 'extraforces'], path + 'pv1-'))
    u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_2', pv2, dump_every, ['forces', 'extraforces'], path + 'pv2-'))
    ###############

    u.run(10002, dt=1.0*dt)

    