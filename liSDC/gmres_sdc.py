import numpy as np
import progressbar

from sdc import sdc
from particleIntegrator import Integrator, Particle
from mygmres import mygmres


class gmres_sdc(sdc):

  def __init__(self, dt, nodes, kiter, pic, nsteps, type, integrate=True, a = None):

    super(gmres_sdc, self).__init__(dt, nodes, kiter, nsteps, type, integrate)
    self.nodes   = nodes
    self.nsweeps = 0
    self.nquads  = 0
    self.kpic    = pic
    self.dim = self.nodes*2*np.shape(self.P.pos0)[0]

    self.Qdelta_E   = np.zeros((self.nodes,self.nodes))
    self.Qdelta_E_2 = np.zeros((self.nodes,self.nodes))
    self.Qdelta_T   = np.zeros((self.nodes,self.nodes))
    
    if a is None:
      for m in range(self.nodes):
        self.Qdelta_E[m,0:m] = self.delta_tau[1:m+1]
      
      for m in range(self.nodes):
        for n in range(self.nodes):
          self.Qdelta_E_2[m,n] = 0.5*self.Qdelta_E[m,n]*self.Qdelta_E[m,n]
      
      for m in range(self.nodes):
        self.Qdelta_T[m, m]  = self.delta_tau[m]
        for n in range(m):
          self.Qdelta_T[m, n] += self.delta_tau[n] + self.delta_tau[n + 1]
      self.Qdelta_T = 0.5 * self.Qdelta_T
    else:
      self.Qdelta_E[1,0]   = a[0]
      self.Qdelta_E[2,0]   = a[1]
      self.Qdelta_E[2,1]   = a[2]
      self.Qdelta_E_2[1,0] = a[3]
      self.Qdelta_E_2[2,0] = a[4]
      self.Qdelta_E_2[2,1] = a[5]
      self.Qdelta_T[0,0]   = a[6]
      self.Qdelta_T[1,0]   = a[7]
      self.Qdelta_T[1,1]   = a[8]
      self.Qdelta_T[2,0]   = a[9]
      self.Qdelta_T[2,1]   = a[10]
      self.Qdelta_T[2,2]   = a[11]
    
  def unpack(self, u):
    x = np.reshape(u[0:self.dim/2], (3,self.nodes), order='F')
    v = np.reshape(u[self.dim/2:self.dim], (3,self.nodes), order='F')
    return x, v   

  def pack(self, x, v):
    x = np.reshape(x, (self.dim/2,), order='F')
    v = np.reshape(v, (self.dim/2,), order='F')
    return np.hstack((x,v))

  def boris_trick(self, x_new, v_old, c_i, coeff):
    E      = 0.5*coeff*self.P.E(x_new) * self.P.alpha
    t      = coeff*self.P.B(x_new) * self.P.alpha
    s      = 2.0*t/(1.0 + np.dot(t, t))
    v_min  = v_old + E + 0.5*c_i
    v_star = v_min + np.cross(v_min, t)
    v_plu  = v_min + np.cross(v_star, s)
    return v_plu + E + 0.5*c_i
  
  '''
  Returns solution of (Id - Q_delta*F)*U = b
  '''
  def sweep(self, u):

    self.nsweeps += 1

    x, v = self.unpack(u)
    xnew = np.zeros((3,self.nodes))
    vnew = np.zeros((3,self.nodes))
    
    for j in range(self.nodes):

      xnew[:,j] = x[:,j]
      b         = np.zeros(3)

      for i in range(j):
        xnew[:,j] += self.Qdelta_E[j,i]*vnew[:,i]
        xnew[:,j] += self.Qdelta_E_2[j,i]*self.F(xnew[:,i],vnew[:,i])
        b         += self.Qdelta_T[j,i]*self.F(xnew[:,i],vnew[:,i])

      coeff     = self.Qdelta_T[j,j]
      bprime    = b - coeff*np.cross(v[:,j], self.P.B(xnew[:,j]))  * self.P.alpha
      vnew[:,j] = self.boris_trick(xnew[:,j], v[:,j], bprime, coeff)
    
    return self.pack(xnew, vnew)

  '''
  Returns solution of (Id - Q_delta*F_lin)*U = b
  '''
  def sweep_linear(self, u, x0):

    self.nsweeps += 1
    x, v = self.unpack(u)
    xnew = np.zeros((3,self.nodes))
    vnew = np.zeros((3,self.nodes))
    
    for j in range(self.nodes):
      xnew[:,j] = x[:,j]
      b = np.zeros(3)
      for i in range(j):
        xnew[:,j] += self.delta_tau[i+1]*vnew[:,i]
        xnew[:,j] += 0.5*self.delta_tau[i+1]**2*self.F(x0[:,i],vnew[:,i])
        b         += 0.5*(self.delta_tau[i] + self.delta_tau[i+1])*self.F(x0[:,i],vnew[:,i])
  
      coeff     = 0.5*self.delta_tau[j]
      bprime    = b - coeff*np.cross(v[:,j], self.P.B(x0[:,j]))  * self.P.alpha
      vnew[:,j] = self.boris_trick(x0[:,j], v[:,j], bprime, coeff)
    
    return self.pack(xnew, vnew)  
  
  '''
  Returns Q*F(U) if x0 is not provided. Otherwise returns Q*F_lin(U) with F_lin being F linearised around x0.
  '''
  def quad(self, u, x0 = None):
    self.nquads += 1
    x, v = self.unpack(u)
    if x0 is None:
      self.updateIntegrals(x, v)
    else:
      self.updateIntegrals(x0, v)
      
    Qx = np.zeros((3,self.nodes))
    for uu in range(self.nodes):
      for nn in range(uu + 1):
        Qx[:,uu] += self.IV_m_mp1[:,nn]
        
    Qv = np.zeros((3,self.nodes))
    for uu in range(self.nodes):
      for nn in range(uu + 1):
        Qv[:,uu] += self.I_m_mp1[:,nn]
        
    return self.pack(Qx, Qv)

  def run(self, bfield, outnames, params):
    # Buffers for k+1 and k solution at integer step
    x_old = np.zeros((3,self.nodes))
    v_old = np.zeros((3,self.nodes))
    x_new = np.zeros((3,self.nodes))
    v_new = np.zeros((3,self.nodes))
    
    # Store values for Bref subroutine:
    xu = np.zeros([self.nodes,3])
    vu = np.zeros([self.nodes,3])    
    p = Particle(self.nodes, self.nsteps)
    
    bar = progressbar.ProgressBar()
    for nn in bar(range(self.nsteps)):
      '''
      Predictor: solve (Id - Q_delta *F) u^1 = u0 + 0.5*Fu0 + v00 (see notes for definitions)
      '''
      x0 = np.kron(np.ones(self.nodes).transpose(), self.positions[:,nn])
      v0 = np.kron(np.ones(self.nodes).transpose(), self.velocities[:,nn])
      u0 = self.pack(x0, v0)

      Fu0 = self.F(self.positions[:,nn], self.velocities[:,nn])
      Fu0 = np.kron(np.ones(self.nodes).transpose(), Fu0)
      Fu0 = self.pack( self.delta_tau[0]**2*Fu0, self.delta_tau[0]*Fu0)
      v00 = self.pack( self.delta_tau[0]*np.kron(np.ones(self.nodes).transpose(), self.velocities[:,nn]), np.zeros(self.dim/2))

      u_old = self.sweep(u0 + 0.5*Fu0 + v00)
      x_old, v_old = self.unpack(u_old)

      '''
      SDC iteration
      '''  
      if self.kpic > 0:
        precond         = lambda b : self.sweep_linear(b, x_old)
        quadrature      = lambda u : u - self.quad(u, x_old)
      else:
        precond         = lambda b : self.sweep(b)
        quadrature      = lambda u : u - self.quad(u)
      u_new, self.stats['residuals'][0:self.kiter,nn] = mygmres(quadrature, u0, u_old, self.kiter, precond)

      '''
      Picard iteration to adjust for nonlinearity
      '''
      for kk in range(self.kpic):
        u_new = u0 + self.quad(u_new)
      x_old, v_old    = self.unpack(u_new)
      
      for i in range(self.nodes):
        xu[i,:] = x_old[:,i]
        vu[i,:] = v_old[:,i]
        
      '''
      Prepare next time step
      '''
      # Compute residual after final iteration
      self.stats['residuals'][self.kiter,nn] = self.getResiduals(self.positions[:,nn], self.velocities[:,nn], x_old, v_old)
        
      self.positions[:,nn+1], self.velocities[:,nn+1] = self.finalUpdateStep(x_old, v_old, self.positions[:,nn], self.velocities[:,nn])
      self.stats['energy_errors'][nn] = self.getEnergyError(self.positions[:,nn+1], self.velocities[:,nn+1])
      
      xu[self.nodes-1,:] = self.positions[:,nn+1]
      vu[self.nodes-1,:] = self.velocities[:,nn+1]

      if p.P.Bound(xu[self.nodes-1,:]):
          break
      if bfield:
        p.GetBrefPoint(nn, self.nodes, self.dt, xu, vu, self.delta_tau)
    p.GetStanDev(self.dt, outnames['deviation'].fullname, params)
      
    return self.positions, self.velocities, self.stats
