#!/usr/bin/env python

#my test for 2body SW

import mirheo as mir
import pint

if __name__ == "__main__":
    ureg = pint.UnitRegistry()

    # Define Mirheo's unit system.
    ureg.define('mirL = 1 nm')
    ureg.define('mirT = 1 fs')
    ureg.define('mirM = 1e-23 g')
    mir.set_unit_registry(ureg)


    dt = ureg('10 fs')            		   # timestep
    rc = ureg('0.432 nm')                  # cutoff radius
    number_density = ureg('33.32819504701638 nm^-3')   # 31.25 paper & 33.32772149 real Water density #33.327677048150235

    ranks  = (1, 1, 1)
    domain = (ureg('20.0 nm'), ureg('20.0 nm'), ureg('20.0 nm'))

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')
    
    pv = mir.ParticleVectors.ParticleVector('pv', mass = ureg('18.015/6.02214076e23 g'))
    ic = mir.InitialConditions.Uniform(number_density)
    u.registerParticleVector(pv, ic)

    # Create and register sw interaction with specific parameters and cutoff radius
    
    sw2 = mir.Interactions.Pairwise('sw', rc, kind="SW", epsilon=ureg('6.189/6.02214076e23 kcal'), sigma=ureg('0.23925 nm'), A=7.049556277, B=0.6022245584)
    u.registerInteraction(sw2)
    u.setInteraction(sw2, pv, pv)

    sw3 = mir.Interactions.Triplewise('sw3', rc, kind='SW3', lambda_=23.15, epsilon=ureg('6.189/6.02214076e23 kcal'), theta=1.9106332362490186, gamma=1.2, sigma=ureg('0.23925 nm'))
    u.registerInteraction(sw3)
    u.setInteraction(sw3, pv, pv, pv)

    """
    dummy = mir.Interactions.Triplewise('interaction', rc=rc, kind='Dummy', epsilon=ureg('6.189/6.02214076e23 kcal'))
    u.registerInteraction(dummy)
    u.setInteraction(dummy, pv, pv, pv)
    """
    ##############################
    # Stage 1. stabilize with Minimize
    ##############################

    minimize = mir.Integrators.Minimize('int_minimize', max_displacement=ureg('0.01 nm'))
    u.registerIntegrator(minimize)
    u.setIntegrator(minimize, pv)

    plugin_stats = mir.Plugins.createStats('stats', every=200)
    u.registerPlugins(plugin_stats)
    u.run(1002, dt=dt)

    u.deregisterPlugins(plugin_stats)
    u.deregisterIntegrator(minimize)
    del plugin_stats
    del minimize

    ##############################
    # Stage 2. vv with very small timesteps
    ##############################
    
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)
    
    #Berendsen Thermostat for holding temp const							
    u.registerPlugins(mir.Plugins.createBerendsenThermostat('thermostat', [pv], T=ureg('298 K'), tau=ureg('1.0 ps')))

    plugin_stats = mir.Plugins.createStats('stats', every=100)
    u.registerPlugins(plugin_stats)

    u.run(10002, dt=0.1*dt)
    
    u.deregisterPlugins(plugin_stats)
    del plugin_stats

    ##############################
    # Stage 3. rdf run (like paper)
    ##############################
    
    # Write some simulation statistics on the screen every 2 time steps
    u.registerPlugins(mir.Plugins.createStats('stats', every=1000))

    #rdf output
    u.registerPlugins(mir.Plugins.createRdf('rdf', pv, max_dist=ureg('4.0 nm'), nbins=1000, basename="rdf_2/pv-", every=100))

    # Dump particle data (for debugging)
    dump_every = 100
    u.registerPlugins(mir.Plugins.createDumpParticles('part_dump', pv, dump_every, [], 'h5/pv-'))


    u.run(1002, dt=dt)
    u.saveSnapshot('snapshot1')
