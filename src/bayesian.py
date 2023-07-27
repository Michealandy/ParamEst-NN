from qutip import *
import numpy as np
from scipy import optimize

from scipy import stats
from scipy.linalg import expm
from scipy.stats import norm
from time import time

from numpy import exp, cos, cosh, sqrt

class model:
    def __init__(self,H,c_ops,J,psi0,with_eigvals=True):
        self.H = H
        self.c_ops = c_ops
        self.J = J.full()
        self.L = liouvillian(H,c_ops)
        self.psi0 = psi0
        self.Lhat = self.L-J
        self.rho0 = operator_to_vector(ket2dm(psi0)).full()
        
        if with_eigvals==True:
            eigen_hat=self.Lhat.eigenstates()
            h2 = len(eigen_hat[1])
            E = np.transpose(np.reshape(
                np.array([eigen_hat[1][i].full() for i in range(h2)]), 
                (h2,h2)))
            Einv = np.linalg.inv(E)
            self.Ehat = E
            self.Einvhat = Einv
            self.eigs_hat = eigen_hat[0]
        else:
            self.Ehat = None
            self.Einvhat = None
            self.eigs_hat = None


def to_time_delay(array):
    return np.concatenate((np.asarray([array[0]]),np.diff(array)))
def to_time_delay_matrix(matrix):
    return np.concatenate((np.reshape(matrix[:,0],(matrix.shape[0],1)),np.diff(matrix)),axis=1)

def gen_MC_jumps_from_model(model=model, tfin = float, ntraj=int, tlistexp = None):
    'tlistexp: If given, [vector of times where expectation values should be computed]'
    H = model.H
    c_ops = model.c_ops
    psi0 = model.psi0

    options = Options()
 
    if len(tlistexp)==0:
        options.store_final_state = False
        tlist = [0,tfin]
        sol=mcsolve(H,psi0,tlist,c_ops,[],ntraj=ntraj,progress_bar=False,options=options)
        times = sol.col_times
        taus = []
        for time_jumps in times:
            if len(time_jumps)>0:
                taus.append(to_time_delay(time_jumps))
            else:
                taus.append(np.array([]))
        return taus
    else:
        options.store_states = True
        tlist = tlistexp
        sigma = c_ops[0]
        sol=mcsolve(H,psi0,tlist,c_ops,[],ntraj=ntraj,progress_bar=False,options=options)
        times = sol.col_times
        taus = []
        for time_jumps in times:
            if len(time_jumps)>0:
                taus.append(to_time_delay(time_jumps))
            else:
                taus.append(np.array([]))
        statestraj = sol.states
        pop_array = np.asarray([[expect(sigma.dag()*sigma, state) for state in statetime] for statetime in statestraj])
        return taus, pop_array
    


def generate_clicks_Model(model,nsaltosMC=48):
    '''Generate a set of nsaltosMC time delays. Used for training (fixed number of clicks)'''
    H = model.H
    c_ops = model.c_ops
    psi0 = model.psi0
           
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

            if len(taus_list)>= nsaltosMC:
                flag = 1
    taus = np.array(taus_list)
    return taus[:nsaltosMC]

    



def likelihood_fun_model(data: np.array, r: float = 5.0, model = model,tfin=-1,method="spectral") -> np.array:
    """Likelihood function that will tell us how probable the data is given our model with parameters delta.

    Args:
        data (np.array): The trajectory data of the time delays
        r (float, optional): A normalization factor. Defaults to 5..
        model(class model): Instance of model class containing Hamiltonian, Liouvillian and jump operators
        tfin = If Not -1, we establish a final time interval with no clicks. If -1, the final time is given by the final jump.
    Returns:
        np.array: The likelihood function for each observation (we compute it after each jump)
    """
    tau_list = data
    t_total = np.sum(tau_list)
    njumps = len(tau_list)
  
    J = model.J
    psi0=model.psi0
    Lhat = model.Lhat
   
    hilbert = psi0.shape[0]
    rho0 = model.rho0
    rhoC = rho0
    rhoC_t = []
    
    if method=='spectral':
        # Spectral decomposition of (L-J)
        E=model.Ehat 
        Einv=model.Einvhat
        EinvTr = np.transpose(Einv)
        eigvals = model.eigs_hat

               
        Uops = [  E@(np.transpose((exp(tau*eigvals))*EinvTr))       for tau in tau_list]
    
       
        # Now we need to use numpy arrays    
            
        for jump_idx  in range(njumps):
            renorm = tau_list[jump_idx]*r
            rhoC=J@Uops[jump_idx]@rhoC
            rhoC = renorm*rhoC
            rhoC_t.append(rhoC)

        # If the simulation set a final time (not a fixed # of clicks):
        if tfin!= -1:
            tau_fin = tfin-t_total
            renorm = tau_fin*r
            Ufin = E@(np.transpose((exp(tau_fin*eigvals))*EinvTr)) 
            rhoC=Ufin@rhoC
            rhoC = renorm*rhoC
            rhoC_t.append(rhoC)
        
        likelihoodTime = np.array([np.reshape(rhoCsel,(hilbert,hilbert)).trace() for rhoCsel in rhoC_t])   

    if method=='direct':
        Uops = [(tau*Lhat).expm() for tau in tau_list]

        for jump_idx  in range(njumps):
            renorm = tau_list[jump_idx]*r
            rhoC=J*Uops[jump_idx]*rhoC
            rhoC = renorm*rhoC
            rhoC_t.append(rhoC)

        # If the simulation set a final time (not a fixed # of clicks):
        if tfin!= -1:
            tau_fin = tfin-t_total
            renorm = tau_fin*r
            Ufin = (tau_fin*Lhat).expm()
            rhoC=Ufin*rhoC
            rhoC = renorm*rhoC
            rhoC_t.append(rhoC)

        likelihoodTime = np.array([vector_to_operator(rhoCsel).tr() for rhoCsel in rhoC_t])   

    return likelihoodTime



def likelihood_fun(delta: float, data: np.array, r: float = 5.0,gamma: float = 1., omega: float = 1.) -> np.array:
    """Likelihood function that will tell us how probable the data is given our model with parameters delta.

    Args:
        delta (float): The parameter of the system. In our case this is a scalar representing the detuning
        data (np.array): The trajectory data of the time delays
        r (float, optional): A normalization factor. Defaults to 5..

    Returns:
        np.array: The likelihood function for each observation (we compute it after each jump)
    """
    tau_list = data

    njumps = len(tau_list)
    psi0=basis(2,0)
    a=destroy(2)
    c_ops=[]
    c_ops.append(np.sqrt(gamma)*a)
    H=delta*a.dag()*a+omega*a+omega*a.dag()
    L = liouvillian(H,c_ops)
    J = gamma*sprepost(a,a.dag())
    Lhat = L-J
    Uops = [(tau*Lhat).expm() for tau in tau_list]
    rho0 = operator_to_vector(ket2dm(psi0))
    rhoC = rho0

    rhoC_t = []
    for jump_idx  in range(njumps):
        renorm = tau_list[jump_idx]*r
        rhoC=J*Uops[jump_idx]*rhoC
        rhoC = renorm*rhoC
        rhoC_t.append(rhoC)

    likelihoodTime = np.array([vector_to_operator(rhoCsel).tr() for rhoCsel in rhoC_t])   


    return likelihoodTime[-1]

def gen_prob_delta(taus,model_list,tfin=-1):
    likelihood_time_delta = np.array([likelihood_fun_model(data= taus, model = model,tfin=tfin,method="spectral") for model in model_list])
    prob_delta = likelihood_time_delta/np.sum(likelihood_time_delta,axis=0)
    return prob_delta

# Same as before, just a more general name
def gen_prob_model_list(taus,model_list,tfin=-1):
    likelihood_time_delta = np.array([likelihood_fun_model(data= taus, model = model,tfin=tfin,method="spectral") for model in model_list])
    prob_delta = likelihood_time_delta/np.sum(likelihood_time_delta,axis=0)
    return prob_delta

def w(tau,Omega,Delta):
   return 8*exp(-tau/2)*Omega**2/sqrt(-64*Omega**2+(1+4*Delta**2+16*Omega**2)**2)*(-cos(tau*sqrt(-1+4*Delta**2+16*Omega**2+sqrt(-64*Omega**2+(1+4*Delta**2+16*Omega**2)**2))/(2*sqrt(2)))
    + cosh(tau*sqrt(1-4*Delta**2-16*Omega**2+sqrt(-64*Omega**2+(1+4*Delta**2+16*Omega**2)**2))/(2*sqrt(2))))

    

def log_likelihood_fun_Analytical(delta: float, data: np.array,gamma: float = 1., omega: float = 1.) -> np.array:
    """Likelihood function that will tell us how probable the data is given our model with parameters delta.

    Args:
        delta (float): The parameter of the system. In our case this is a scalar representing the detuning
        data (np.array): The trajectory data of the time delays
        

    Returns:
        np.array: The likelihood function for each observation (we compute it after each jump)
    """
    tau_list = data
    wlist = np.array([w(tau,omega,delta) for tau in tau_list])
    log_likelihood = np.sum(np.log(wlist))+1e-300

    return log_likelihood

def neg_log_likelihood_fun_Analytical(delta: float, data: np.array,gamma: float = 1., omega: float = 1.) -> np.array:
    return -log_likelihood_fun_Analytical(delta, data,gamma, omega)

def likelihood_fun_Analytical(delta: float, data: np.array, r: float = 5.0,gamma: float = 1., omega: float = 1.) -> np.array:
    """Likelihood function that will tell us how probable the data is given our model with parameters delta.

    Args:
        delta (float): The parameter of the system. In our case this is a scalar representing the detuning
        data (np.array): The trajectory data of the time delays
    Returns:
        np.array: The likelihood function for each observation (we compute it after each jump)
    """


    log_likelihood =  log_likelihood_fun_Analytical(delta, data, gamma, omega)

    return np.exp(log_likelihood)


def negative_log_likelihood(x, data):
    delta, = x
    return -np.log(likelihood_fun(delta, data)+1e-300)

def find_min(tau_list):
    return np.abs(optimize.minimize(neg_log_likelihood_fun_Analytical, 1.0, tol=1e-8, args=(tau_list), method="Nelder-Mead").x[0])

def find_min_full_output(tau_list):
    return optimize.minimize(neg_log_likelihood_fun_Analytical, 1.0, tol=1e-8, args=(tau_list), method="Nelder-Mead")

def find_median(deltaList,probList):
    return deltaList[np.argmin(np.abs(np.cumsum(probList)-0.5))]

def get_posterior(tau_list):
    deltaBayesListFine = np.linspace(0.,4,500)
    likelihood_Fine=np.array([likelihood_fun_Analytical(delta,tau_list) for delta in deltaBayesListFine])
    probFine = likelihood_Fine/np.sum(likelihood_Fine)
    
    return probFine

def get_likelihood(tau_list):
    deltaBayesListFine = np.linspace(0.,4,500)
    likelihood_Fine=np.array([likelihood_fun_Analytical(delta,tau_list) for delta in deltaBayesListFine])    
    return likelihood_Fine

def get_estimates(tau_list):
    deltaBayesListFine = np.linspace(0.,4,500)
    likelihood_Fine=np.array([likelihood_fun_Analytical(delta,tau_list) for delta in deltaBayesListFine])
    probFine = likelihood_Fine/np.sum(likelihood_Fine)
    deltaMean = np.dot(deltaBayesListFine,probFine)
    deltaMedian = find_median(deltaBayesListFine,probFine)
    deltaMax = deltaBayesListFine[np.argmax(probFine)]

    return deltaMean, deltaMedian,deltaMax, probFine
        


def prob_no_click(delta,tau,omega=1.):
    return np.real(
        ((-1/8*1j)*
  (exp(((2 + np.emath.sqrt(2)*np.emath.sqrt(1 - 4*delta**2 - 16*omega**2 + 
           np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))))*
       tau)/4)*((-8*1j)*np.emath.sqrt(2)*delta**3 + (2*1j)*np.emath.sqrt(2)*delta*
      (-1 - 16*omega**2 + np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 
         8*delta**2*(1 + 16*omega**2))) + np.emath.sqrt(1 - 4*delta**2 - 16*omega**2 + 
       np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))) + 
     4*delta**2*np.emath.sqrt(1 - 4*delta**2 - 16*omega**2 + 
        np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))) - 
     16*omega**2*np.emath.sqrt(1 - 4*delta**2 - 16*omega**2 + 
        np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))) - 
     np.emath.sqrt((16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))*
       (1 - 4*delta**2 - 16*omega**2 + np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 
          8*delta**2*(1 + 16*omega**2))))) + 
   exp(((2 + np.emath.sqrt(2)*np.emath.sqrt(1 - 4*delta**2 - 16*omega**2 + 
           np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))) + 
        (2*1j)*np.emath.sqrt(2)*np.emath.sqrt(-1 + 4*delta**2 + 16*omega**2 + 
           np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))))*
       tau)/4)*((-8*1j)*np.emath.sqrt(2)*delta**3 + (2*1j)*np.emath.sqrt(2)*delta*
      (-1 - 16*omega**2 + np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 
         8*delta**2*(1 + 16*omega**2))) - np.emath.sqrt(1 - 4*delta**2 - 16*omega**2 + 
       np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))) - 
     4*delta**2*np.emath.sqrt(1 - 4*delta**2 - 16*omega**2 + 
        np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))) + 
     16*omega**2*np.emath.sqrt(1 - 4*delta**2 - 16*omega**2 + 
        np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))) + 
     np.emath.sqrt((16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))*
       (1 - 4*delta**2 - 16*omega**2 + np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 
          8*delta**2*(1 + 16*omega**2))))) + 
   exp(((2 + 1j*np.emath.sqrt(2)*np.emath.sqrt(-1 + 4*delta**2 + 16*omega**2 + 
           np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))))*
       tau)/4)*((8*1j)*np.emath.sqrt(2)*delta**3 + (2*1j)*np.emath.sqrt(2)*delta*
      (1 + 16*omega**2 + np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 
         8*delta**2*(1 + 16*omega**2))) - (4*1j)*delta**2*
      np.emath.sqrt(-1 + 4*delta**2 + 16*omega**2 + np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 
          8*delta**2*(1 + 16*omega**2))) - 
     1j*(np.emath.sqrt(-1 + 4*delta**2 + 16*omega**2 + np.emath.sqrt(16*delta**4 + 
           (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))) - 
       16*omega**2*np.emath.sqrt(-1 + 4*delta**2 + 16*omega**2 + 
          np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))) + 
       np.emath.sqrt((16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))*
         (-1 + 4*delta**2 + 16*omega**2 + np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 
            8*delta**2*(1 + 16*omega**2)))))) + 
   exp(((2 + 2*np.emath.sqrt(2)*np.emath.sqrt(1 - 4*delta**2 - 16*omega**2 + 
           np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))) + 
        1j*np.emath.sqrt(2)*np.emath.sqrt(-1 + 4*delta**2 + 16*omega**2 + 
           np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))))*
       tau)/4)*((8*1j)*np.emath.sqrt(2)*delta**3 + (2*1j)*np.emath.sqrt(2)*delta*
      (1 + 16*omega**2 + np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 
         8*delta**2*(1 + 16*omega**2))) + (4*1j)*delta**2*
      np.emath.sqrt(-1 + 4*delta**2 + 16*omega**2 + np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 
          8*delta**2*(1 + 16*omega**2))) + 
     1j*(np.emath.sqrt(-1 + 4*delta**2 + 16*omega**2 + np.emath.sqrt(16*delta**4 + 
           (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))) - 
       16*omega**2*np.emath.sqrt(-1 + 4*delta**2 + 16*omega**2 + 
          np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))) + 
       np.emath.sqrt((16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2))*
         (-1 + 4*delta**2 + 16*omega**2 + np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 
            8*delta**2*(1 + 16*omega**2))))))))/
 (np.emath.sqrt(2)*delta*
  exp(((4 + np.emath.sqrt(2)*np.emath.sqrt(1 - 4*delta**2 - 16*omega**2 + 
         np.emath.sqrt(-64*omega**2 + (1 + 4*delta**2 + 16*omega**2)**2)) + 
      1j*np.emath.sqrt(2)*np.emath.sqrt(-1 + 4*delta**2 + 16*omega**2 + 
         np.emath.sqrt(-64*omega**2 + (1 + 4*delta**2 + 16*omega**2)**2)))*tau)/4)*
  np.emath.sqrt(16*delta**4 + (1 - 16*omega**2)**2 + 8*delta**2*(1 + 16*omega**2)))
    )