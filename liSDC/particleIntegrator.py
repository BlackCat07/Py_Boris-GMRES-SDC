from __future__ import division
from scipy import interpolate

import progressbar
import numpy as np
import math as ma
import copy

from pySDC.gauss_lobatto  import CollGaussLobatto
from problem_mt           import problem


class DataOut():
    
    def __init__ (self, name, method, nd, it, pic):
        self.name = name
        self.method = method
        self.fullname = name + "_" + method +"_n" + str(nd) + "i" + str(it) + "p" + str(pic)


class Integrator(object):

    def __init__(self, nodes):
        
        # Arrays with dim M x 3, where M - nodes, 3 - coordinates
        self.x = np.zeros([nodes, 3])
        self.x_ = np.zeros([nodes, 3])
        self.v = np.zeros([nodes, 3])
        self.v_ = np.zeros([nodes, 3])
        self.xend = np.zeros([nodes, 3])
        self.vend = np.zeros([nodes, 3])
        self.x0 = np.zeros([nodes, 3])
        self.v0 = np.zeros([nodes, 3])
        
        # Set initial values for x and v:
        self.x0[:, :] = copy.deepcopy(self.P.pos0[:])
        self.v0[:, :] = copy.deepcopy(self.P.vel0[:])
        self.x[0, :] = copy.deepcopy(self.P.pos0)
        self.v[0, :] = copy.deepcopy(self.P.vel0)

        # Initialize matrices:
        self.S = np.zeros([nodes, nodes])
        self.Q = np.zeros([nodes, nodes])
        self.ST = np.zeros([nodes, nodes])
        self.SQ = np.zeros([nodes, nodes])
        self.Sx = np.zeros([nodes, nodes])
        self.QQ = np.zeros([nodes, nodes])
        self.dt = np.zeros([nodes])
        self.wi = np.zeros([nodes])
        
    def ReadData(self, file_name):
        with open(file_name, 'r') as data:
            x = []
            y = []
            for line in data:
                p = line.split()
                x.append(float(p[0]))
                y.append(float(p[1]))
        return x, y
        
    def RHSeval(self, params, name, method_id):
        pre = 'rhs_'
        nd = params['nodes']
        it = params['itera']
        pic = params['picit']
        
        
        if   method_id == 'BrefGMRes':
            coeff = nd + (nd - 1) + pic * (nd - 1) + (nd - 1)
        elif method_id == 'BrefSDC':
            coeff = nd + it * (nd - 1)
        elif method_id == 'BrefBrs':
            coeff = 1
        else:
            print "From RHSeval: wrong method ID"
            
        x, y = self.ReadData(name)
        newdata = open(pre + name, "w")
        
        for i in range(len(x)):
            newdata.write('%e %e\n' % (coeff * params['tend'] * self.P.om_c/x[i], y[i]))
        

    def GetMatrices(self, nodes, left, right):

        L = CollGaussLobatto(nodes, left, right)
        QI = np.zeros(L.Qmat.shape)
        QE = np.zeros(L.Qmat.shape)

        for m in range(L.num_nodes + 1):
            QI[m, 1:m + 1] = L.delta_m[0:m]

        for m in range(L.num_nodes + 1):
            QE[m, 0:m] = L.delta_m[0:m]

        # trapezoidal rule
        QT = 1.0 / 2.0 * (QI + QE)

        # Qx as in the paper
        Qx = np.dot(QE, QT) + 1 / 2 * QE * QE

        Sx = np.zeros(np.shape(L.Qmat))
        ST = np.zeros(np.shape(L.Qmat))
        S = np.zeros(np.shape(L.Qmat))

        # fill-in node-to-node matrices
        Sx[0, :] = Qx[0, :]
        ST[0, :] = QT[0, :]
        S[0, :] = L.Qmat[0, :]
        for m in range(L.num_nodes):
            Sx[m + 1, :] = Qx[m + 1, :] - Qx[m, :]
            ST[m + 1, :] = QT[m + 1, :] - QT[m, :]
            S[m + 1, :] = L.Qmat[m + 1, :] - L.Qmat[m, :]

        Q = L.Qmat

        # SQ via dot-product, could also be done via QQ
        SQ = np.dot(S, L.Qmat)

        # QQ-matrix via product of Q
        QQ = np.dot(L.Qmat, L.Qmat)

        # Slice matrices:
        S = S[1:, 1:]
        Q = Q[1:, 1:]
        ST = ST[1:, 1:]
        SQ = SQ[1:, 1:]
        Sx = Sx[1:, 1:]
        QQ = QQ[1:, 1:]

        dt = L.delta_m
        wi = L.weights

        return [S, ST, SQ, Sx, QQ, Q, dt, wi]

    # Implementation of 39(a,b) equations,
    # Journal of Computational Physics 295 (2015) 456-474
    def Sweeper(self, nd, it, x, v, x0, v0, x_, v_, keyinteg):

        S = self.S
        SQ = self.SQ
        Sx = self.Sx
        dt = self.dt
        Sum = np.zeros(3)
        ck = np.zeros(3)

        # Iteration cycle:
        for j in range(it + 1):

            # Matrix product:
            SF = np.dot(S, self.F(x_, v_))
            SQF = np.dot(SQ, self.F(x_, v_))

            # Sweep cycle:
            for m in range(nd-1):

                Sum[:] = 0
                # Get Sx*(f(x,v) - f(x_,v_)):
                for i in range(0, m + 1):
                    Sum[:] += Sx[m+1, i]*(self.F(x, v)[i, :] - self.F(x_, v_)[i, :])

                # Get c_k value:
                ck[:] = -dt[m+1]*(self.F(x_, v_)[m+1, :] + self.F(x_, v_)[m, :])/2.0 + SF[m+1, :]

                # Get x_m+1
                x[m+1, :] = x[m, :] + dt[m+1]*v0[m+1, :] + Sum[:] + SQF[m+1, :]

                # Call Boris method
                v[m + 1, :] = self.BorisMethodMod(ck, x[m, :], x[m+1, :], v[m, :], dt[m+1])
                
                # Get v_m+1
                v[m + 1, :] = v[m, :] + SF[m+1, :] + dt[m+1]*(self.F(x, v)[m + 1, :]
                                                                - self.F(x_, v_)[m+1, :])/2.0\
                                                   + dt[m+1]*(self.F(x, v)[m, :]
                                                                - self.F(x_, v_)[m, :])/2.0

            # Copy data after each iteration (x^k+1 -> x^k)
            x_[:, :] = x[:, :]
            v_[:, :] = v[:, :]
        
        # Integration procedure:
        integratorOption = getattr(self, keyinteg)(nd, x, v, x0, v0)
        
    # Only for non-stg Boris
    def SweeperNoVelUp(self, nd, it, x, v, x0, v0, x_, v_, keyinteg):

        S = self.S
        SQ = self.SQ
        Sx = self.Sx
        dt = self.dt
        Sum = np.zeros(3)
        ck = np.zeros(3)

        # Iteration cycle:
        for j in range(it + 1):

            # Matrix product:
            SF = np.dot(S, self.F(x_, v_))
            SQF = np.dot(SQ, self.F(x_, v_))

            # Sweep cycle:
            for m in range(nd-1):

                Sum[:] = 0
                # Get Sx*(f(x,v) - f(x_,v_)):
                for i in range(0, m + 1):
                    Sum[:] += Sx[m+1, i]*(self.F(x, v)[i, :] - self.F(x_, v_)[i, :])

                # Get c_k value:
                ck[:] = -dt[m+1]*(self.F(x_, v_)[m+1, :] + self.F(x_, v_)[m, :])/2.0 + SF[m+1, :]

                # Get x_m+1
                x[m+1, :] = x[m, :] + dt[m+1]*v0[m+1, :] + Sum[:] + SQF[m+1, :]

                # Call Boris method
                v[m + 1, :] = self.BorisMethodMod(ck, x[m, :], x[m+1, :], v[m, :], dt[m+1])

            # Copy data after each iteration (x^k+1 -> x^k)
            x_[:, :] = x[:, :]
            v_[:, :] = v[:, :]
        
        # Integration procedure:
        integratorOption = getattr(self, keyinteg)(nd, x, v, x0, v0)
    
    def UpdateDat(self, nd):
        
        # Get full velocity at current position t[i]:
        self.vmod = np.sqrt(self.vend[nd - 1, 0]**2 + self.vend[nd - 1, 1]**2 + self.vend[nd - 1, 2]**2)

        # Denotation for simplicity:
        v = self.vend[nd - 1, :]
        B = self.P.B(self.xend[nd-1, :])

        # Get Cos(phi), v^B at current position t[i]:
        self.phi = np.dot(v, B)/(self.vmod * self.P.Bmod(self.xend[nd-1, :]))
        
        # Set new init values for v0 and x0:
        self.x0[:, :] = copy.deepcopy(self.xend[nd - 1, :])
        self.v0[:, :] = copy.deepcopy(self.vend[nd - 1, :])

        # Reset to zero:
        self.x[:, :] = 0.0
        self.v[:, :] = 0.0
        self.x[0, :] = self.x0[nd - 1, :]
        self.v[0, :] = self.v0[nd - 1, :]
        self.x_[:, :] = 0.0
        self.v_[:, :] = 0.0

    def IntegrateOn(self, nd, x, v, x0, v0):
        Q = self.Q
        QQ = self.QQ
        for u in range(nd):
            self.xend[u, :] = x0[u, :] + np.dot(QQ, self.F(x, v))[u, :] + np.dot(Q, v0)[u, :]
            self.vend[u, :] = v0[u, :] + np.dot(Q, self.F(x, v))[u, :]

    def IntegrateOff(self, nd, x, v, x0, v0):        
        self.xend[:, :] = self.x[:, :]
        self.vend[:, :] = self.v[:, :]

    def F(self, x, v):
        rhs = np.zeros([len(self.x), len(self.x[0])])
        for i in range(len(x)):
            rhs[i, :] = self.P.alpha*(self.P.E(x[i, :]) + np.cross(v[i, :], self.P.B(x[i, :])))
        return rhs
    
    def BorisMethodMod(self, c, x_old, x_new, v_old, dt):

        Bmean = self.P.B(x_new)
        Emean = 0.5*(self.P.E(x_old) + self.P.E(x_new))
        c += self.P.alpha * np.cross(v_old, self.P.B(x_old) - self.P.B(x_new))*dt/2.0
        t = self.P.alpha*Bmean*dt/2.0
        tsq = np.dot(t, t)
        s = 2*t/(1 + tsq)
        v_min = v_old + self.P.alpha*Emean*dt/2.0 + c/2.0
        v_htc = v_min + np.cross(v_min, t)
        v_pls = v_min + np.cross(v_htc, s) + c/2.0
        vn = v_pls + self.P.alpha*Emean*dt/2.0
        return vn
        
    def GetBrefPoint(self, j, nd, step, x, v, dt):
        
        xref = np.zeros(3)
        
        # Get module velocity at 1st and last nodes:
        vmodO = np.sqrt(v[0, 0]*v[0, 0] + v[0, 1]*v[0, 1] + v[0, 2]*v[0, 2])
        vmodN = np.sqrt(v[nd-1, 0]*v[nd-1, 0] + v[nd-1, 1]*v[nd-1, 1] + v[nd-1, 2]*v[nd-1, 2])
        # Get Magnetic field at 1st and last nodes:
        Bold = self.P.B(x[0, :])
        Bnew = self.P.B(x[nd-1, :])
        cosphi_o = np.dot(v[0, :], Bold) / (vmodO * self.P.Bmod(x[0, :]))
        cosphi_n = np.dot(v[nd-1, :], Bnew) / (vmodN * self.P.Bmod(x[nd-1, :]))
        # Get parallel velocity at 1st and last nodes:
        vparold = vmodO*cosphi_o
        vparnew = vmodN*cosphi_n

        # Check for reflection:
        if vparnew*vparold/abs(vparnew*vparold) < 1.0:

            delt = 0.0
            vparnd = np.zeros(nd)
            timend = np.zeros(nd)
            
            # get vpar at all nodes:
            for i in range(nd):
                delt += dt[i]
                timend[i] = j * step + delt

                Bnd = self.P.B(x[i, :])
                vmodnd = np.sqrt(v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2)
                cosphind = np.dot(v[i, :], Bnd) / (vmodnd * self.P.Bmod(x[i, :]))
                vparnd[i] = vmodnd*cosphind
                
            # Calculate time when {vpar = 0}:
            tm = self.BisectionMethod(timend, vparnd, nd)
            
            # Calculate x components at {t = tm}:
            for k in range(3):
                xref[k] = self.LPoly(tm, timend, x[:, k], nd)
            self.breflist.append(self.P.Bmod(xref))

    # Bisection method for root finding:
    def BisectionMethod(self, time, vpar, nd):
        
        a = time[0]
        b = time[nd-1]
        tol = 1.e-12
        itr = 0

        while abs(self.LPoly((a + b)/2.0,time,vpar,nd)) > tol:

            mid = (a + b)/2.0
            if self.LPoly(mid,time,vpar,nd) == 0:
                timeref = mid
                break
            elif self.LPoly(a,time,vpar,nd)*self.LPoly(mid,time,vpar,nd) < 0:
                b = mid
            else:
                a = mid

            itr += 1
            if itr > 50:
                break                
        timeref = mid
        return timeref
    
    # Returns Lagrange polynomial interpolation:
    def LPoly(self, x, xarray, farray, nd):
        
        if x < xarray[0] or x > xarray[nd-1]:
            sys.exit("variable out of range...")
        f = 0.0
        
        # Interpolation polynomial in the Lagrange form:
        for i in range(nd):
            f += farray[i]*self.Lcoeff(nd, i, x, xarray)
        return f
    
    # Returns Lagrange l(x) coeff. value:
    def Lcoeff(self, numpnt, j, x, xarray): 
        l = 1.0
        for i in range(numpnt):
            if i != j:
                l *= (x - xarray[i])/(xarray[j] - xarray[i])
        return l

    # Calculates standart deviation of data:
    def GetStanDev(self, step, name, params):
    
        if params['case'] == 'nonadiabatic':

            # Load reference data:
            time, bfield = self.ReadData("magrefE13N7I5")
            summ = 0
            
            # Check the number of reflection points:
            if len(self.breflist) == len(bfield):            
                for idx, item in enumerate(self.breflist):
                    summ += (bfield[idx] - self.breflist[idx])**2
            
                self.stndev = np.sqrt(summ/len(self.breflist))       
                dev = open(name, "a")
                dev.write('%e %e\n' % (step * self.P.om_c, self.stndev))
                dev.close()
                print "STD", self.stndev
            else:
                print "From GetStanDev: too many reflection points"
                print "Exiting..."
        
        elif params['case'] == 'adiabatic':
        
            summ = 0
            # Check the number of reflection points:
            if len(self.breflist) > 5: 
                for idx, item in enumerate(self.breflist):
                    summ += (self.Bref - self.breflist[idx])**2
                
                self.stndev = np.sqrt(summ/len(self.breflist))
                dev = open(name, "a")
                dev.write('%e %e\n' % (step * self.P.om_c, self.stndev))
                dev.close()
                print "Reflections:", len(self.breflist)
                print "STD", self.stndev
            else:
                print "From GetStanDev: not enough reflections"
                print "Exiting..."
                
        else:
            print "Please use 'nonadiabatic' or 'adiabatic' key only"
            print "Exiting..."


class Particle(Integrator):
    
    def __init__(self, nodes, nsteps):
        
        # Define the problem
        self.P = problem()
        
        # Arrays: magnetic moment, position and velocity:
        self.mu = np.zeros([int(nsteps)])
        self.nrg = np.zeros([int(nsteps)])
        self.crd = np.zeros([int(nsteps), 3])
        self.vls = np.zeros([int(nsteps), 3])
        
        # NOTE! 'UpdateDat' returns Cos(phi)
        self.phi = 0.0
        
        self.breflist = []
        self.vmod = 0.0
        self.stndev = 0.0
        
        # Kinetic initial energy:
        self.Ekin0 = np.dot(self.P.vel0, self.P.vel0)/2.0
        
        # Initial pitch angle and full velocity:
        self.vmod0 = np.sqrt(np.dot(self.P.vel0, self.P.vel0))
        self.phi0 = ma.acos(np.dot(self.P.vel0, self.P.B(self.P.pos0))/(self.vmod0 * self.P.Bmod(self.P.pos0)))

        if self.P.name == 'MT':
            # Magnetic field on start position:
            self.Bzer = self.P.Bmod(self.P.pos0)
            # Magnetic field on coils:
            self.Bmax = self.P.Bmod(self.P.CoilPos)
            # Theoretical value of magnetic field at ref. point:
            self.Bref = self.Bzer*self.vmod0**2/(self.vmod0**2 - (self.vmod0*ma.cos(self.phi0))**2)
            # Minimal pitch angle:
            self.phiMin = ma.asin(np.sqrt(self.Bzer/self.Bmax))
            self.mu0 = (self.vmod0*ma.sin(self.phi0))**2/self.Bzer
            
            # Check for condition for pitch angle:
            if self.phi0 < self.phiMin:
                print "Phi min:", self.phiMin
                print "Your Phi:", self.phi0
                sys.exit("Out of trap. Change initial velosity")
        
        super(Particle, self).__init__(nodes)
