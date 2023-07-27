from qutip import *
import numpy as np


def to_time_delay(array):
    return np.concatenate((np.asarray([array[0]]),np.diff(array)))

def generate_clicks_Params(param,njumpsMC=48,gamma=1.):
    '''Generate a set of nsaltosMC time delays given a delta'''
    delta = param[0]
    omega = param[1]
    a = destroy(2)
    H=delta*a.dag()*a+omega*a+omega*a.dag()
           
    t0 = 0
    factor = 1.2
    tf=factor*njumpsMC*(4*delta**2+8*omega**2+gamma**2)/(4*gamma*omega**2)
    tlist=[t0,tf] 

    options = Options()
    options.store_final_state = True
    
    flag = 0
    psi0=basis(2,0)
    c_ops=[]
    c_ops.append(np.sqrt(gamma)*a)

    taus_list = []

    tshift = t0
    while flag ==0:
            
            sol=mcsolve(H,psi0,tlist,c_ops,[],ntraj=1,progress_bar=False,options=options)
            
            #My new initial psi is given by the final psi
            psi0 = sol.states[-1,0]
            times = tshift + sol.col_times[0]
            tshift = times[-1]
            taus = list(to_time_delay(times))
            taus_list += taus

            if len(taus_list)>= njumpsMC:
                flag = 1
    taus = np.array(taus_list)
    return taus[:njumpsMC]


def generate_clicks_3LS(params,nsaltosMC=48):
    [delta,kappa,omega] = params 
    psi0=basis(3,0); a=destroy(3); c_ops=[]; c_ops.append(np.sqrt(kappa)*a)
    H=delta*a.dag()*a+omega*a+omega*a.dag()
    J = kappa*sprepost(a,a.dag())          
    t0 = 0
    factor = 1.2

    jumpOp = c_ops[0]
    rho_ss = steadystate(H,c_ops)
    n_expect = expect(jumpOp.dag()*jumpOp, rho_ss)

    tf=factor*nsaltosMC/n_expect
    tlist=[t0,tf] 

    options = Options()
    options.store_final_state = True
    
    flag = 0
    times_list = []

    tshift = t0
    while flag ==0:
            
            sol=mcsolve(H,psi0,tlist,c_ops,[],ntraj=1,progress_bar=False,options=options)
            
            #My new initial psi is given by the final psi
            psi0 = sol.states[-1,0]
            times = list(tshift + sol.col_times[0])
            tshift = times[-1]
            #taus = list(to_time_delay(times))
            times_list += times

            if len(times_list)>= nsaltosMC:
                flag = 1
    times_array = np.asarray(times_list)
    taus = to_time_delay(times_array)
    return taus[:nsaltosMC]
