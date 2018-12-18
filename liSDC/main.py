from __future__ import division
from controller import Controller
import numpy as np


def main():
    
    # Some input names:
    outkeys = dict()
    outkeys['deviation']    = 'stndev'
    outkeys['trajectory']   = 'traj'
    outkeys['convergence']  = 'conv'
    outkeys['totalenergy']  = 'ener'
    outkeys['positions']    = 'pos'
    outkeys['velocities']   = 'vel'
    
    # Input parameters:
    params = dict()
    params['case'] =  'nonadiabatic'
    params['tend'] =  16.0
    params['itera'] = 0
    params['picit'] = 0
    
    
    '''
    Standard deviation of the magnetic field value at all reflection points
    '''
    # Boris:
    params['nodes'] = 2
    params['itera'] = 0
    params['picit'] = 0
    Controller.BrefBrs(params, outkeys)
    
    # GMRES-SDC(1,1) - 3 nodes:
    params['nodes'] = 3
    params['picit'] = 1
    params['itera'] = 1
    Controller.BrefGMRes(params, outkeys, 'IntegrateOn')


    '''
    Long-time energy error
    '''
    
    params['tend'] =  4800.0    # Runtime
    nstp = 3840000              # Number of steps
    
    # Boris:
    params['nodes'] = 2
    params['itera'] = 0
    Controller.EnergyBrs(params, outkeys, nstp)
    
    # GMRES-SDC(2,2) - 3 nodes:
    params['nodes'] = 3
    params['itera'] = 2
    params['picit'] = 2
    Controller.EnergyGMRes(params, outkeys, 'IntegrateOn', nstp)
    
    
if __name__ == "__main__":
    main()
