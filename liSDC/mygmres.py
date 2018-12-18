import numpy as np

'''
From C. T. Kelley, "Iterative Methods for Linear and Nonlinear Equations", SIAM Philadelphia 1995.
'''
def mygmres(A, b, x, m, Minv):

  norm = lambda x : np.linalg.norm(x)

  n = np.shape(b)[0]
  
  residuals = np.zeros(m)
  
  b_norm = norm(b)
  if b_norm == 0:
    b_norm = 1.0

  r      = Minv(b - A(x)) ### Total cost: m+1 matrix-vector products
  r_norm = norm(r)
  error  = r_norm/b_norm

  # initialise buffers
  s = np.zeros(m)
  c = np.zeros(m)

  # Initialise basis with residual  vector
  Q = np.zeros((n,m+1))
  Q[:,0] = r*r_norm**(-1)

  # Right hand side for minimisation problem
  e1    = np.zeros(n+1)
  e1[0] = 1.0
  g     = r_norm*e1

  H = np.zeros((m+1,m))

  for n in range(m):
  
    Q[:,n+1], H[0:n+2,n] = arnoldi(A, Minv, Q, n)

    '''
    Lines (f) and (g) in Algorithm 3.5.1 on p. 45 in Kelley
    '''
    if n>0:
      H[0:n+1,n] = apply_qn(H[0:n+1,n], c, s, n)

    nu = np.sqrt(H[n,n]**2 + H[n+1,n]**2)
    c[n] = H[n,n]/nu
    s[n] = -H[n+1,n]/nu
    H[n,n] = c[n]*H[n,n] - s[n]*H[n+1,n]
    H[n+1,n] = 0.0

    # Apply single rotation to g
    g[n+1] = s[n]*g[n]
    g[n]   *= c[n]

    residuals[n] = abs(g[n+1]/b_norm)
  
  # Compute solution
  y = np.linalg.solve(H[0:m,0:m], g[0:m])
  return x + Q[:,0:m].dot(y), residuals

'''
Lines (b) and (c) from Algorithm 3.5.1 on p. 45 in Kelley
'''
def arnoldi(A, Minv, Q, n):
    h = np.zeros(n+2)
    v = Minv(A(Q[:,n])) # !!!!
    for j in range(n+1):
      h[j] = np.dot(Q[:,j], v)
      v -= h[j]*Q[:,j]
    h[n+1] = np.linalg.norm(v)
    v *= 1.0/h[n+1]
    return v, h

'''
See Eq. (3.13) on p. 44 in Kelley
'''
def apply_qn(h, c, s, n):
  for i in range(n):
    temp   = c[i]*h[i] - s[i]*h[i+1]
    h[i+1] = s[i]*h[i] + c[i]*h[i+1]
    h[i]   = temp
  return h
