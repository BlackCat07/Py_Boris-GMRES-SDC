import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

from particleIntegrator import Integrator, Particle
from problem_mt import problem


class sdc(object):

  def __init__(self, dt, nodes, kiter, nsteps, type, integrate=True):

    self.dt     = dt
    self.kiter  = kiter
    self.nsteps = nsteps
    self.nodes  = nodes
    
    g = Particle(nodes, nsteps)
    self.S, dum1, dum2, dum3, dum4, dum5, self.delta_tau, self.q = g.GetMatrices(nodes, 0.0, dt)
    
    self.integrate = integrate
    if type=='legendre':
      if not self.integrate:
        print("Overiding integrate=False... for Legendre nodes, integrate needs to be set to True")
        self.integrate = True
    if (self.kiter==0) and (self.integrate):
      print("Setting kiter=0 but integrate=True will NOT reproduce the standard Boris integrator")
    
    # Define the problem
    self.P = problem()

    self.stats = {}
    self.stats['residuals']     = np.zeros((kiter+1,nsteps))
    self.stats['increments']    = np.zeros((kiter,nsteps))
    self.stats['energy_errors'] = np.zeros(nsteps)

    # Buffers to store approximations at time steps
    self.positions       = np.zeros((3,nsteps+1))
    self.velocities      = np.zeros((3,nsteps+1))
    self.positions[:,0]  = self.P.pos0
    self.velocities[:,0] = self.P.vel0

    self.stats['exact_energy'] = self.P.Energy(self.positions[:,0], self.velocities[:,0])

    self.stats['errors']      = np.zeros((2,nsteps+1))
    self.stats['errors'][0,0] = 0.0
    self.stats['errors'][1,0] = 0.0

  def F(self,x,v):
    return self.P.alpha * (self.P.E(x) + np.cross(v, self.P.B(x)))
  
  def getEnergyError(self, x, v):
    return np.abs( self.P.Energy(x, v) - self.stats['exact_energy'])/np.abs(self.stats['exact_energy'])

  def updateIntegrals(self, x, v):
    F = np.zeros((3,self.nodes))
    for jj in range(self.nodes):
      F[:,jj] = self.F(x[:,jj], v[:,jj]) # F[:,jj] = F_j

    # Set integral terms to zero
    self.I_m_mp1    = np.zeros((3,self.nodes))
    self.IV_m_mp1   = np.zeros((3,self.nodes))
    for jj in range(self.nodes):
      # self.I_m_mp1[:,jj] equals I_j^j+1
      for kk in range(self.nodes):
        self.I_m_mp1[:,jj]    += self.S[jj,kk]*F[:,kk]
        self.IV_m_mp1[:,jj]   += self.S[jj,kk]*v[:,kk]

  def getResiduals(self, x0, v0, x, v):
    
    self.updateIntegrals(x, v)
        
    # Compute residuals
    res_x = np.zeros((3,self.nodes))
    res_v = np.zeros((3,self.nodes))

    res_v[:,0] = v[:,0] - v0 - self.I_m_mp1[:,0]
    for i in range(1,self.nodes):
      res_v[:,i] = v[:,i] - v[:,i-1] - self.I_m_mp1[:,i]

    res_x[:,0] = x[:,0] - x0 - self.IV_m_mp1[:,0]
    for i in range(1,self.nodes):
      res_x[:,i] = x[:,i] - x[:,i-1] - self.IV_m_mp1[:,i]
    
    return max(np.linalg.norm(res_v, np.inf), np.linalg.norm(res_x, np.inf))

  def finalUpdateStep(self, x, v, x0, v0):
    if self.integrate:
      F = np.zeros((3,self.nodes))
      for jj in range(self.nodes):
        F[:,jj] = self.F(x[:,jj], v[:,jj])
      x_final = np.copy(x0)
      v_final = np.copy(v0)
      for jj in range(self.nodes):
        x_final += self.q[jj]*v[:,jj]
        v_final += self.q[jj]*F[:,jj]
    else:
      x_final = x[:,2]
      v_final = v[:,2]
    return x_final, v_final
