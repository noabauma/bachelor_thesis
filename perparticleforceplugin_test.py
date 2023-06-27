import mirheo as mir
import numpy as np
import h5py
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main():
    if rank == 0:
        position = np.array([[5.0, 5.0, 5.0],
                            [5.0, 5.5, 5.0],
                            [5.5, 5.0, 5.0],
                            [5.5, 5.5, 5.0],
                            [6.0, 5.0, 5.0],
                            [6.0, 5.5, 5.0]])
        extraforces = np.array([[0.0, 0.0, 200.0],
                            [-300.0, 0.0, 0.0],
                            [ 0.0, 0.0, 0.0],
                            [ 0.0, 0.0, 0.0],
                            [ 100.0, 0.0, 0.0],
                            [ 100.0, 0.0, 0.0]])
        id = np.reshape(np.arange(6), (6,1))
        velocity = np.zeros((6,3))

        os.remove("h5_c/pv.PV-00000.h5")
        f = h5py.File("h5_c/pv.PV-00000.h5", "a")
        dset = f.create_dataset("position", (6,3), dtype=np.float64)
        dset[...] = position
        dset = f.create_dataset("velocity", (6,3), dtype=np.float64)
        dset[...] = velocity
        dset = f.create_dataset("id", (6,1), dtype=np.int)
        dset[...] = id
        dset = f.create_dataset("extraforces", (6,3), dtype=np.float64)
        dset[...] = extraforces
        print(f.keys())
        f.close()
    

    ranks = (1, 1, 1)
    domain = (10.0, 10.0, 10.0)

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True, checkpoint_every=10)

    pv = mir.ParticleVectors.ParticleVector('pv', mass=1.0)
    ic = mir.InitialConditions.Restart('h5_c')
    u.registerParticleVector(pv, ic)

    u.registerPlugins(mir.Plugins.createAddPerParticleForce("extra_force", pv, "extraforces"))    #my plugin
    
    sw3 = mir.Interactions.Triplewise('interaction', rc=1.0, kind='SW3', lambda_=0.0, epsilon=0.0, theta=1.0, gamma=1.0, sigma=1.0)
    u.registerInteraction(sw3)
    u.setInteraction(sw3, pv, pv, pv)
    
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)
    
    dump_every = 1
    u.registerPlugins(mir.Plugins.createForceSaver('forces', pv))
    u.registerPlugins(mir.Plugins.createDumpParticles('force_dump', pv, dump_every, ["forces", "extraforces"], 'h5_c/sw3-'))

    u.run(5, dt=0.0)

    if rank == 0:
        f_ = h5py.File('h5_c/sw3-00003.h5', 'r')
        print(f_['extraforces'][()])
        print(f_['position'][()])
        print(f_['forces'][()])




main()