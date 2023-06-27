#!/usr/bin/env python

"""Test triplewise forces."""

import mirheo as mir
import pint
import numpy as np
import sys
import h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

np.random.seed(42)

def main():
    if len(sys.argv) < 1 and rank == 0:
        print("needs triplewise(bool) \n")
        sys.exit(0)
    
    triplewise = 1

    ureg = pint.UnitRegistry()

    # Define Mirheo's unit system.
    ureg.define('mirL = 1 nm')
    ureg.define('mirT = 1 fs')
    ureg.define('mirM = 1e-23 g')
    mir.set_unit_registry(ureg)


    #dt = ureg('10 fs')            		   # timestep
    rc = ureg('0.432 nm')                  # cutoff radius
    number_density = ureg('33.328 nm^-3')  # 31.25 normally  
    

    ranks  = (1, 1, 1)
    domain = (ureg('20.0 nm'), ureg('20.0 nm'), ureg('20.0 nm'))

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')

    pv = mir.ParticleVectors.ParticleVector('pv', mass = ureg('18.015/6.02214076e23 g'))

    #ic = mir.InitialConditions.Uniform(number_density)
    ic = mir.InitialConditions.Restart("snapshot2")

    u.registerParticleVector(pv, ic)

    """SW3"""
    if triplewise:
        if rank == 0: 
            print("SW3 test\n", flush=True)
        sw3 = mir.Interactions.Triplewise('sw3', rc, kind='SW3', lambda_=23.15, epsilon=ureg('6.189/6.02214076e23 kcal'), theta=1.910633236, gamma=1.2, sigma=ureg('0.23925 nm'))
        u.registerInteraction(sw3)
        u.setInteraction(sw3, pv, pv, pv)
    
    """SW2"""
    if triplewise:
        if rank == 0: 
            print("SW2 test\n", flush=True)
        sw2 = mir.Interactions.Pairwise('sw', rc, kind="SW", epsilon=ureg('6.189/6.02214076e23 kcal'), sigma=ureg('0.23925 nm'), A=7.049556277, B=0.6022245584)
        u.registerInteraction(sw2)
        u.setInteraction(sw2, pv, pv)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

    #stats for checking
    u.registerPlugins(mir.Plugins.createStats('stats', every=4))

    u.run(10, dt=0)



main()
