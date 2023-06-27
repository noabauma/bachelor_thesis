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


    #load graphene particle positions
    position = np.loadtxt('particle_generators/grs1p.xyz', skiprows=2, dtype=str)

    position = position[:,1:].astype(np.float64)
    n = position.shape[0]
    velocity = np.zeros((n,3))

    shift = np.array([10.0, 10.0, 7.6]) #shift all particles to the center
    for i in range(n):
        position[i,:] += shift

    
    
    dt = ureg('1 fs')            		   # timestep
    rc = ureg('2.5 angstrom')              # cutoff radius

    ranks  = (1, 1, 1)
    domain = (ureg('520.0 angstrom'), ureg('520.0 angstrom'), ureg('15.0 angstrom'))

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')
    
    pv = mir.ParticleVectors.ParticleVector('pv', mass = ureg('12.0107/6.02214076e23 g'))
    ic = mir.InitialConditions.FromArray(pos=position, vel=velocity)
    u.registerParticleVector(pv, ic)

    
    #Interactions
    
    sw2 = mir.Interactions.Pairwise('sw2', rc, kind='SW', epsilon=ureg('1.0 eV'), sigma=ureg('2.025 angstrom'), A=38.39709, B=0.09399)
    u.registerInteraction(sw2)
    u.setInteraction(sw2, pv, pv)
    

    sw3 = mir.Interactions.Triplewise('sw3', rc, kind='SW3', lambda_=41.0, epsilon=ureg('1.0 eV'), theta=2.0943951023931957, gamma=0.55, sigma=ureg('2.025 angstrom'))
    u.registerInteraction(sw3)
    u.setInteraction(sw3, pv, pv, pv)

    """
    dummy = mir.Interactions.Triplewise('interaction', rc=rc, kind='Dummy', epsilon=0.0)
    u.registerInteraction(dummy)
    u.setInteraction(dummy, pv, pv, pv)
    """
    #Integrators
    
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)
    

    ####Plugins####				
    #u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv], T=ureg('10.0 K'), tau=ureg('100.0 fs')))
    u.registerPlugins(mir.Plugins.createStats('stats', every=1000))
    
    ###############
    u.run(1000001, dt=0)
