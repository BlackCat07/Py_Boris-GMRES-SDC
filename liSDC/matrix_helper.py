from computeWeights import computeWeights

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

def getMatrices(dt, type, a = None):

    S, q, delta_tau = computeWeights(type)


    '''
    Generate SDC related matrices
    '''
    Q      = np.zeros((3,3))
    Q[0,:] = S[0,:]
    Q[1,:] = S[0,:] + S[1,:]
    Q[2,:] = S[0,:] + S[1,:] + S[2,:]

    # In case argument is not provided, construct Qdelta_E, Qdelta_E_2 and Qdelta_T normally
    Qdelta_I   = np.zeros((3,3))
    Qdelta_E   = np.zeros((3,3))
    Qdelta_E_2 = np.zeros((3,3))
    Qdelta_T   = np.zeros((3,3))
    if a is None:
      Qdelta_E[:,0] = delta_tau[1]*np.array([0,1,1])
      Qdelta_E[:,1] = delta_tau[2]*np.array([0,0,1])
      Qdelta_I[:,0] = delta_tau[0]*np.array([1,1,1])
      Qdelta_I[:,1] = delta_tau[1]*np.array([0,1,1])
      Qdelta_I[:,2] = delta_tau[2]*np.array([0,0,1])
      Qdelta_T      = 0.5*(Qdelta_E + Qdelta_I)
    else:
      Qdelta_E[1,0]   = a[0]
      Qdelta_E[2,0]   = a[1]
      Qdelta_E[2,1]   = a[2]
      Qdelta_E_2[1,0] = a[3]
      Qdelta_E_2[2,0] = a[4]
      Qdelta_E_2[2,1] = a[5]
      Qdelta_T[0,0]   = a[6]
      Qdelta_T[1,0]   = a[7]
      Qdelta_T[1,1]   = a[8]
      Qdelta_T[2,0]   = a[9]
      Qdelta_T[2,1]   = a[10]
      Qdelta_T[2,2]   = a[11]
     
    Qdelta_E_2 = np.kron( np.multiply(Qdelta_E, Qdelta_E), np.eye(3))
    Qdelta_E_2 *= 0.5*dt**2
    Qdelta_T  = dt*np.kron( Qdelta_T, np.eye(3))
    Qdelta_E  = dt*np.kron(Qdelta_E, np.eye(3))
    Q         = dt*np.kron(Q, np.eye(3))


    '''
    Generate iteration matrices
    '''
    Zero = np.zeros((9,9))
    Id   = np.eye(9)

    QQ1 = np.hstack((Q, Zero))
    QQ2 = np.hstack((Zero, Q))
    QQ  = np.vstack((QQ1, QQ2))

    FF1 = np.hstack((Zero, Id))
    FF2 = np.hstack((E_mat, B_mat))
    FF  = np.vstack((FF1, FF2))

    QQd1     = np.hstack((Qdelta_E, Qdelta_E_2))
    QQd2     = np.hstack((Zero, Qdelta_T))
    QQ_delta = np.vstack((QQd1, QQd2))

    return QQ, QQ_delta, FF
