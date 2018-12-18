from __future__ import division
from scipy import interpolate

import progressbar
import numpy as np

from gmres_sdc import gmres_sdc
from particleIntegrator import Integrator, Particle, DataOut


class Stepsize(object):
    def __init__ (self):
        sn = 8000           # should be integer
        self.st = 7
        self.step = np.zeros(self.st)
        
        # Nonadiabatic case:
        self.step[:] = [32*sn, 16*sn, 8*sn, 4*sn, 2*sn, sn, sn/2]
        
        # Adiabatic case:
        #self.step[:] = [4.E6, 2.E6, 1.E6, 0.5E6, 0.25E6]


class Inform(object):
    @staticmethod
    def Display(nodes, it, RoutineName):
        print ""
        print "Running", RoutineName, "with", nodes, "nodes and", it, "iteration(s)"
    
    @staticmethod
    def storeEnergy(namefile, datanum, energy):
        nfilter = 100
        dat = open(namefile, "w")
        for u in range(datanum):
            if ((u + 1) % nfilter) == 0:
                dat.write('%e %e\n' % (u+1, energy[u]))
        dat.close()
        print("%.14e" % energy[datanum - 1])
        
    @staticmethod
    def storeConverg(namefile, datanum, x, om):
        dat = open(namefile, "w")
        for u in range(datanum):
            ierror = abs(x[0, 0] - x[u + 1, 0])/abs(x[0, 0])
            dat.write('%10e %.16e\n' % (om[u], ierror))
        dat.close()

    @staticmethod
    def storePositions(namefile, datanum, x, om):
        dat = open(namefile, "w")
        for u in range(datanum):
            dat.write('%.10e %.16e %.16e %.16e\n' % (om[u], x[u,0], x[u,1], x[u,2]))
        dat.close()
        
    @staticmethod
    def writeMu(namefile, datanum, dt, y):
        dat = open(namefile, "w")
        for u in range(datanum):
            dat.write('%.10e %.16e\n' % (dt*(u+1), y[u]))
        dat.close()
        
    @staticmethod
    def storeVelocities(namefile, datanum, x, om):
        dat = open(namefile, "w")
        for u in range(datanum):
            dat.write('%.10e %.16e %.16e %.16e\n' % (om[u], x[u,0], x[u,1], x[u,2]))
        dat.close()

class Controller(object):
    
    @staticmethod #tested
    def Trajectory(params, outkeys, keyinteg, numsteps):

        Inform.Display(params['nodes'], params['itera'], 'Trajectory')
        lastpnt = 0
        bar = progressbar.ProgressBar(widgets=[progressbar.Bar(), ' (', progressbar.ETA(), ') ']) 
        outnames = {key: DataOut(   name = outkeys[key], method = 'sdc', nd = params['nodes'], 
                                    it = params['itera'], pic = params['picit']) for key in outkeys}
        
        step = params['tend']/numsteps
        p = Particle(params['nodes'], numsteps)
        p.S, p.ST, p.SQ, p.Sx, p.QQ, p.Q, p.dt, p.wi = p.GetMatrices(params['nodes'], 0.0, step)
        
        bar = progressbar.ProgressBar()
        for i in bar(range(int(numsteps))):
            
            p.Sweeper(params['nodes'], params['itera'], p.x, p.v, p.x0, p.v0, p.x_, p.v_, keyinteg)
            p.UpdateDat(params['nodes'])
            p.crd[i, :] = p.xend[params['nodes'] - 1, :]
            p.vls[i, :] = p.vend[params['nodes'] - 1, :]
            
            p.mu[i] = (p.vmod**2 - (p.vmod*p.phi)**2)/p.P.Bmod(p.xend[params['nodes'] - 1, :])
            
            if p.P.Bound(p.xend[params['nodes'] - 1, :]):
                lastpnt = i
                break

        if lastpnt != 0:
            datanum = lastpnt
        else:
            datanum = numsteps

        dat = open(outnames['trajectory'].fullname, "w")
        for u in range(datanum):
            if ((u + 1) % 100) == 0:
                dat.write('%e %e %e\n' % (p.crd[u, 0], p.crd[u, 1], p.crd[u, 2]))
        dat.close()

    @staticmethod 
    def EnergySDC(params, outkeys, keyinteg, numsteps):
        
        Inform.Display(params['nodes'], params['itera'], 'EnergySDC')
        lastpnt = 0
        bar = progressbar.ProgressBar(widgets=[progressbar.Bar(), ' (', progressbar.ETA(), ') ']) 
        outnames = {key: DataOut(   name = outkeys[key], method = 'sdc', nd = params['nodes'], 
                                    it = params['itera'], pic = params['picit']) for key in outkeys}
        
        step = params['tend']/numsteps
        p = Particle(params['nodes'], numsteps)
        p.S, p.ST, p.SQ, p.Sx, p.QQ, p.Q, p.dt, p.wi = p.GetMatrices(params['nodes'], 0.0, step)
        
        bar = progressbar.ProgressBar()
        for i in bar(range(int(numsteps))):
                
            p.Sweeper(params['nodes'], params['itera'], p.x, p.v, p.x0, p.v0, p.x_, p.v_, keyinteg)
            p.UpdateDat(params['nodes'])
            
            p.crd[i, :] = p.xend[params['nodes'] - 1, :]
            p.vls[i, :] = p.vend[params['nodes'] - 1, :]
            p.nrg[i] = abs(p.Ekin0 - np.dot(p.vls[i,:], p.vls[i,:])/2.0)/p.Ekin0

            if p.P.Bound(p.xend[params['nodes'] - 1, :]):
                lastpnt = i
                break
                
        if lastpnt != 0:
            datanum = lastpnt
        else:
            datanum = numsteps
            
        Inform.storeEnergy(outnames['totalenergy'].fullname, datanum, p.nrg)
        
    @staticmethod 
    def EnergyGMRes(params, outkeys, keyinteg, numsteps):
        
        Inform.Display(params['nodes'], params['itera'], 'EnergyGMRes')
        lastpnt = 0
        bar = progressbar.ProgressBar(widgets=[progressbar.Bar(), ' (', progressbar.ETA(), ') ']) 
        outnames = {key: DataOut(   name = outkeys[key], method = 'gm', nd = params['nodes'], 
                                    it = params['itera'], pic = params['picit']) for key in outkeys}
        
        type      = 'lobatto'
        integrate = True
        bfield    = False
        step = params['tend']/numsteps
        p = Particle(params['nodes'], numsteps)

        gmres = gmres_sdc(dt = step, nodes = params['nodes'], kiter = params['itera'], pic = params['picit'], 
            nsteps = numsteps, type=type, integrate=integrate)
        x_gm, v_gm, stats = gmres.run(bfield = bfield, outnames = outnames, params = params)
        
        Inform.storeEnergy(outnames['totalenergy'].fullname, numsteps, stats['energy_errors'])

    @staticmethod
    def EnergyBrs(params, outkeys, numsteps):
        
        Inform.Display(params['nodes'], params['itera'], 'EnergyBrs')
        lastpnt = 0
        bar = progressbar.ProgressBar(widgets=[progressbar.Bar(), ' (', progressbar.ETA(), ') ']) 
        outnames = {key: DataOut(   name = outkeys[key], method = 'brs', nd = params['nodes'], 
                                    it = params['itera'], pic = params['picit']) for key in outkeys}
        
        step = params['tend']/numsteps
        p = Particle(params['nodes'], numsteps)
        p.S, p.ST, p.SQ, p.Sx, p.QQ, p.Q, p.dt, p.wi = p.GetMatrices(params['nodes'], 0.0, step)

        bar = progressbar.ProgressBar()
        for i in bar(range(int(numsteps))):
                
            p.SweeperNoVelUp(params['nodes'], params['itera'], p.x, p.v, p.x0, p.v0, p.x_, p.v_, 'IntegrateOff')
            p.UpdateDat(params['nodes'])
            
            p.crd[i, :] = p.xend[params['nodes'] - 1, :]
            p.vls[i, :] = p.vend[params['nodes'] - 1, :]
            p.nrg[i] = abs(p.Ekin0 - np.dot(p.vls[i,:], p.vls[i,:])/2.0)/p.Ekin0
            
            if p.P.Bound(p.xend[params['nodes'] - 1, :]):
                lastpnt = i
                break

        if lastpnt != 0:
            datanum = lastpnt
        else:
            datanum = numsteps
            
        Inform.storeEnergy(outnames['totalenergy'].fullname, datanum, p.nrg)
        
    @staticmethod
    def ConvergenceSDC(params, outkeys, keyinteg):
        
        Inform.Display(params['nodes'], params['itera'], 'ConvergenceSDC')
        bar = progressbar.ProgressBar(widgets=[progressbar.Bar(), ' (', progressbar.ETA(), ') '])
        outnames = {key: DataOut(   name = outkeys[key], method = 'sdc', nd = params['nodes'], 
                                    it = params['itera'], pic = params['picit']) for key in outkeys}
        
        n = Stepsize()
        omdt = np.zeros(n.st)
        xpos = np.zeros([n.st, 3])

        for u in range(n.st):
            
            step = params['tend']/n.step[u]
            p = Particle(params['nodes'], n.step[u])
            p.S, p.ST, p.SQ, p.Sx, p.QQ, p.Q, p.dt, p.wi = p.GetMatrices(params['nodes'], 0.0, step)

            bar = progressbar.ProgressBar()
            for i in bar(range(int(n.step[u]))):
                
                p.Sweeper(params['nodes'], params['itera'], p.x, p.v, p.x0, p.v0, p.x_, p.v_, keyinteg)
                p.UpdateDat(params['nodes'])

                if p.P.Bound(p.xend[params['nodes'] - 1, :]):
                    lastpnt = i
                    break
                    
            xpos[u, :] = p.xend[params['nodes']-1, :]
            omdt[u] = step*p.P.om_c
        
        Inform.storePositions(outnames['positions'].fullname, n.st, xpos, omdt)
        Inform.storeConverg(outnames['convergence'].fullname, n.st - 1, xpos, omdt)

    @staticmethod
    def ConvergenceGMRes(params, outkeys, keyinteg):
        
        Inform.Display(params['nodes'], params['itera'], 'ConvergenceGMRes')
        bar = progressbar.ProgressBar(widgets=[progressbar.Bar(), ' (', progressbar.ETA(), ') '])
        outnames = {key: DataOut(   name = outkeys[key], method = 'gm', nd = params['nodes'], 
                                    it = params['itera'], pic = params['picit']) for key in outkeys}

        type      = 'lobatto'
        integrate = True
        bfield = False
        n = Stepsize()
        omdt = np.zeros(n.st)
        xpos = np.zeros([n.st, 3])
        xvel = np.zeros([n.st, 3])

        for u in range(n.st):
            
            step = params['tend']/n.step[u]
            p = Particle(params['nodes'], n.step[u])
            
            gmres = gmres_sdc(dt = step, nodes = params['nodes'], kiter = params['itera'], pic = params['picit'], 
            nsteps = int(n.step[u]), type=type, integrate=integrate)
            x_gm, v_gm, stats = gmres.run(bfield = bfield, outnames = outnames, params = params)
            xpos[u, :] = x_gm[:, int(n.step[u])]
            xvel[u, :] = v_gm[:, int(n.step[u])]
            omdt[u] = step*p.P.om_c
        
        Inform.storePositions(outnames['positions'].fullname, n.st, xpos, omdt)
        Inform.storeConverg(outnames['convergence'].fullname, n.st - 1, xpos, omdt)
    
    @staticmethod
    def ConvergenceBrs(params, outkeys):
        
        Inform.Display(params['nodes'], params['itera'], 'ConvergenceBrs')
        bar = progressbar.ProgressBar(widgets=[progressbar.Bar(), ' (', progressbar.ETA(), ') '])
        outnames = {key: DataOut(   name = outkeys[key], method = 'brs', nd = params['nodes'], 
                                    it = params['itera'], pic = params['picit']) for key in outkeys}
        
        n = Stepsize()
        omdt = np.zeros(n.st)
        xpos = np.zeros([n.st, 3])

        for u in range(n.st):
            
            step = params['tend']/n.step[u]
            p = Particle(params['nodes'], n.step[u])
            p.S, p.ST, p.SQ, p.Sx, p.QQ, p.Q, p.dt, p.wi = p.GetMatrices(params['nodes'], 0.0, step)

            bar = progressbar.ProgressBar()
            for i in bar(range(int(n.step[u]))):
                
                p.SweeperNoVelUp(params['nodes'], params['itera'], p.x, p.v, p.x0, p.v0, p.x_, p.v_, 'IntegrateOff')
                p.UpdateDat(params['nodes'])
                
                if p.P.Bound(p.xend[params['nodes'] - 1, :]):
                    lastpnt = i
                    break
                    
            xpos[u, :] = p.xend[params['nodes']-1, :]
            omdt[u] = step*p.P.om_c
            
        dat = open(outnames['convergence'].fullname, "w")
        for u in range(n.st - 1):
            ierror = abs(xpos[u, 0] - xpos[u+1, 0])/abs(xpos[u, 0])
            dat.write('%e %.16e\n' % (omdt[u], ierror))
        dat.close()
    
    @staticmethod
    def BrefSDC(params, outkeys, keyinteg):
        
        Inform.Display(params['nodes'], params['itera'], 'BrefSDC')
        bar = progressbar.ProgressBar(widgets=[progressbar.Bar(), ' (', progressbar.ETA(), ') '])
        outnames = {key: DataOut(   name = outkeys[key], method = 'sdc', nd = params['nodes'], 
                                    it = params['itera'], pic = params['picit']) for key in outkeys}
        
        n = Stepsize()
        
        for u in range(n.st):
            
            step = params['tend']/n.step[u]
            p = Particle(params['nodes'], n.step[u])
            p.S, p.ST, p.SQ, p.Sx, p.QQ, p.Q, p.dt, p.wi = p.GetMatrices(params['nodes'], 0.0, step)

            bar = progressbar.ProgressBar()
            for i in bar(range(int(n.step[u]))):

                p.Sweeper(params['nodes'], params['itera'], p.x, p.v, p.x0, p.v0, p.x_, p.v_, keyinteg)
                p.UpdateDat(params['nodes'])
                
                if p.P.Bound(p.xend[params['nodes'] - 1, :]):
                    lastpnt = i
                    break
                
                p.GetBrefPoint(i, params['nodes'], step, p.xend, p.vend, p.dt)
            p.GetStanDev(step, outnames['deviation'].fullname, params)
        
        p.RHSeval(params, outnames['deviation'].fullname, 'BrefSDC')
    
    @staticmethod
    def BrefGMRes(params, outkeys, keyinteg):
        
        Inform.Display(params['nodes'], params['itera'], 'BrefGMRes')
        bar = progressbar.ProgressBar(widgets=[progressbar.Bar(), ' (', progressbar.ETA(), ') '])
        outnames = {key: DataOut(   name = outkeys[key], method = 'gm', nd = params['nodes'], 
                                    it = params['itera'], pic = params['picit']) for key in outkeys}

        type      = 'lobatto'
        integrate = True
        bfield    = True
        n = Stepsize()

        for u in range(n.st):
            
            step = params['tend']/n.step[u]
            p = Particle(params['nodes'], n.step[u])
            
            gmres = gmres_sdc(dt = step, nodes = params['nodes'], kiter = params['itera'], pic = params['picit'], 
            nsteps = int(n.step[u]), type=type, integrate=integrate)
            x_gm, v_gm, stats = gmres.run(bfield = bfield, outnames = outnames, params = params)
            
        p.RHSeval(params, outnames['deviation'].fullname, 'BrefGMRes')
        
    @staticmethod
    def BrefBrs(params, outkeys):

        Inform.Display(params['nodes'], params['itera'], 'BrefBrs')
        bar = progressbar.ProgressBar(widgets=[progressbar.Bar(), ' (', progressbar.ETA(), ') '])
        outnames = {key: DataOut(   name = outkeys[key], method = 'brs', nd = params['nodes'], 
                                    it = params['itera'], pic = params['picit']) for key in outkeys}
        
        n = Stepsize()
        
        for u in range(n.st):
            
            step = params['tend']/n.step[u]
            p = Particle(params['nodes'], n.step[u])
            p.S, p.ST, p.SQ, p.Sx, p.QQ, p.Q, p.dt, p.wi = p.GetMatrices(params['nodes'], 0.0, step)

            bar = progressbar.ProgressBar()
            for i in bar(range(int(n.step[u]))):
                
                p.SweeperNoVelUp(params['nodes'], params['itera'], p.x, p.v, p.x0, p.v0, p.x_, p.v_, 'IntegrateOff')
                p.UpdateDat(params['nodes'])
                
                if p.P.Bound(p.xend[params['nodes'] - 1, :]):
                    lastpnt = i
                    break
                
                p.GetBrefPoint(i, params['nodes'], step, p.xend, p.vend, p.dt)
        
            p.GetStanDev(step, outnames['deviation'].fullname, params)
            
        p.RHSeval(params, outnames['deviation'].fullname, 'BrefBrs')

