import sys
from tqdm import tqdm
import numpy as np
import qutip as qt
from dataclasses import dataclass
from typing import Optional

# Class structure for passing around lots of params
@dataclass
class ObjPackage:
    rho_set      : Optional[np.ndarray] = None
    H_coupled_mat: Optional[np.ndarray] = None
    c_evecs      : Optional[np.ndarray] = None
    H_qubit_mat  : Optional[np.ndarray] = None
    q_evals      : Optional[np.ndarray] = None
    q_evecs      : Optional[np.ndarray] = None
    H_res_mat    : Optional[np.ndarray] = None
    r_evals      : Optional[np.ndarray] = None
    r_evecs      : Optional[np.ndarray] = None
    H_cm_mat     : Optional[np.ndarray] = None
    cm_evals     : Optional[np.ndarray] = None
    cm_evecs     : Optional[np.ndarray] = None
    qdim         : Optional[int] = None
    rdim         : Optional[int] = None
    cdim         : Optional[int] = None
    get_full     : Optional[bool] = False

def get_expectation_vals(pkg: ObjPackage):
    # Unpacking params
    H_coupled = pkg.H_coupled_mat
    qdim      = pkg.qdim
    rdim      = pkg.rdim
    cdim      = pkg.cdim
    N_qr      = pkg.qdim * pkg.rdim
    q_evecs   = pkg.q_evecs
    c_evecs   = pkg.c_evecs

    # Lists for storing data
    P_list = []
    N_list = []
    
    # Construct qubit symmetry operators
    I_q = np.eye(qdim)
    P_q = np.diag((-1)**np.arange(qdim)).astype(complex) # Resonator parity operator
    N_q      = np.diag(np.arange(qdim)).astype(complex)     # Qubit number operator
    
    # Construct resonator symmetry operators
    I_r = np.eye(rdim)
    P_r = np.diag((-1)**np.arange(rdim)).astype(complex) # Resonator parity operator
    N_r = np.diag(np.arange(rdim)).astype(complex)       # Resonator number operator

    # Construct chain symmetry operators
    I_c      = np.eye(cdim)
    P_c      = np.diag((-1)**np.arange(cdim)).astype(complex) # Chain parity operator
    N_c      = np.diag(np.arange(cdim)).astype(complex)         # Chain number operator
    N_c_full = np.kron(np.kron(I_q, I_r), N_c) 

    # Construct total symmetry operators
    I_t = np.kron(I_q, I_r) # Entirely unneccessary
    P_t = np.kron(P_q, P_r)
    P_t = np.kron(P_t, P_c)
    N_t = np.kron(N_q, I_r) + np.kron(I_q, N_r)
    N_t = np.kron(N_t, I_c) + np.kron(I_t, N_c)

    # Find commutators for each subspace
    #for n_c in tqdm(range(cdim), ascii=True, desc='checkin\' sectors'):
        # Find each individual chain mode occupation number sector
    #    H_nc         = H_coupled.reshape(N_qr, cdim, N_qr, cdim)[:, n_c, :, n_c]
        # Compute commutators
    #    P_commutator = H_nc @ P_t - P_t @ H_nc
    #    N_commutator = H_nc @ N_t - N_t @ H_nc
    #    P_list.append(P_commutator)
    #    N_list.append(N_commutator)

    P_list  = []
    N_list  = []
    Nc_list = []
    for vec in tqdm(c_evecs.T, ascii=True, desc="checkin\' vecs"):
        OLP  = vec.conj().T @ P_t @ vec
        OLN  = vec.conj().T @ N_t @ vec
        OLNc = vec.conj().T @ N_c_full @ vec 
        P_list.append(OLP)
        N_list.append(OLN)
        Nc_list.append(OLNc)

    return P_list, N_list, Nc_list


    