import numpy as np

class problem(object):

    def __init__(self):
        self.name = 'MT'
        
        # Nonadiabatic case:
        self.pos0 = np.array([1.0, 0.0, 0.0])
        self.vel0 = np.array([100.0, 0.0, 50.0])
        self.z0 = 16.0
        self.om_c = 400.0
        
        '''
        # Simple trajectory:
        self.pos0 = np.array([5.25, 5.25, 0.0])
        self.vel0 = np.array([100.0, 0.0, 50.0])
        self.z0 = 8.0
        self.om_c = 200.0
        '''
        '''
        # Adiabatic case:
        self.pos0 = np.array([1.0, 0.5, 0.0])
        self.vel0 = np.array([100.0, 0.0, 50.0])
        self.z0 = 200.0
        self.om_c = 2000.0
        '''
        
        self.alpha = 1.0
        self.CoilPos = np.array([0.0, 0.0, self.z0])
    
    def B(self, x):
        B = np.zeros(3)
        B0 = self.om_c/self.alpha
        B[0] = -B0*x[0]*x[2]/self.z0**2
        B[1] = -B0*x[1]*x[2]/self.z0**2
        B[2] =  B0*(1 + x[2]**2/self.z0**2)
        return B
        
    def Bmod(self, x):
        Bmod = np.sqrt(self.B(x)[0]*self.B(x)[0] 
            + self.B(x)[1]*self.B(x)[1] 
            + self.B(x)[2]*self.B(x)[2])
        return Bmod

    def E(self, x):
        E = np.zeros(3)
        E[0] = 0.0
        E[1] = 0.0
        E[2] = 0.0
        return E
    
    def Energy(self, x, v):
        return 0.5*np.dot(v,v)
        
    def Bound(self, x):
        return abs(x[2]) > self.z0
