import sys
from tqdm import tqdm
import numpy as np
from scipy.linalg import eigh  
from scipy.sparse.linalg import eigsh  
#from transmon_coupled import *
from fluxonium_coupled import *
import qutip as qt
import matplotlib.pyplot as plt
import scqubits as scq
from scipy.linalg import cosm
from dataclasses import dataclass

# TBD: Rename PCA to something more accurate (overlap matrix)

def oscillator_hamiltonian(omega_r, dim):
    # Create annihilation operator
    a = np.zeros((dim, dim), dtype=complex)
    
    for n in range(1, dim):
        a[n-1, n] = np.sqrt(n)
        
    adag = a.conj().T
    num  = adag @ a
    H    = omega_r * (num)
    return H, a, adag

def fast_ptrace_qubit(c_evecs, qdim, rdim):
    """
    Compute reduced qubit density matrices for each eigenvector
    """
    # Find number of eigenstates
    num_eigs = c_evecs.shape[0] # shape (num_eigs, qdim*rdim)
    # Reshape to (num_eigs, qdim, rdim)
    psi = c_evecs.reshape(num_eigs, qdim, rdim)
    # Compute rho = psi @ psi^† over resonator index
    rho = np.einsum('nqr,npr->npq', psi, psi.conj()) # shape (num_eigs, qdim, qdim)

    return rho

def get_objs(params, qubit_type, H_full=None, update_flux=False):
    if (not update_flux):
        # Check device type, get full Hilbert space/qubit subspace
        if qubit_type == 'fluxonium':
            H_full = Coupled_Fluxonium(params)
            qubit = H_full.fluxonium
        #if qubit_type == 'transmon':
        #    H_full = Coupled_Transmon(params)
        #    qubit = H_full.transmon
    else:
        flux = params['flux'][0]
        H_full.update_flux(flux)
        if qubit_type == 'fluxonium':
            qubit = H_full.fluxonium
        #if qubit_type == 'transmon':
        #    qubit = H_full.transmon
    # Get hilbert space data
    hs = H_full.hs
    # Get resonator subspace/dimension
    resonator = H_full.resonator
    qdim = qubit.truncated_dim
    rdim = resonator.truncated_dim
    
    # Get all the eigenvalues/vectors/density matrices
    H_coupled_mat = hs.hamiltonian().full()
    H_qubit_mat = qubit.hamiltonian()
    H_res_mat, a, adag = oscillator_hamiltonian(H_full.f_r, rdim)
    q_evals, q_evecs = qubit.eigensys(qdim)
    r_evals, r_evecs = resonator.eigensys(rdim)
    c_evals, qc_evecs = hs.eigensys(qdim*rdim)
    c_evecs = np.asarray([vec.full().T[0] for vec in qc_evecs])
    rho_set = fast_ptrace_qubit(c_evecs, qdim, rdim)
    
    # Nice tuple for moving things around
    dat_package = (rho_set, H_coupled_mat, c_evecs, 
                   H_qubit_mat, q_evals, q_evecs, 
                   H_res_mat, r_evals, r_evecs, 
                   qdim, rdim)
    
    return dat_package, H_full

def branch_analysis(rho_set, H_coupled_mat, c_evecs, 
                    H_qubit_mat, q_evals, q_evecs, 
                    H_res_mat, r_evals, r_evecs, 
                    qdim, rdim, flux_loop=False):
    """
    Inputs:
      - H_coupled_mat    : (N,N) numpy array, N = qdim*rdim
      - H_qubit_mat      : (qdim, qdim) numpy array (bare qubit H)
      - q_evals, q_evecs : bare qubit eigenvectors/eigenvalues
      - H_res_mat        : (rdim, rdim) numpy array (bare resonator H)
      - r_evals, r_evecs : bare resonator eigenvectors/eigenvalues
      - qdim, rdim       : dimension of qubit/resonator subspaces
      
    Returns:
      - params      : dictionary containing inputs for sim
      - data        : dictionary containing all data produced in sim
      - params_list : conveniently repacked params for plotting
      - data_list   : conveniently repacked data for plotting
      - branches    : dictionary containing all branch info (almost identical to data)
    """
    
    """""""""
    Pt. 0: Another Function Definition (index assignment)
    """""""""
    
    def ladder_levels(vecs, c_evecs, b_adag, new_vecs=False):
        """
        Vectorized computation of PCA overlaps between vecs and cvecs,
        optionally applying ladder operator to vecs.
        """
        
        if new_vecs:
            vecs = [b_adag @ vec for vec in vecs]  # update vecs

        # Convert lists of Qobj to NumPy arrays
        vecs_array  = np.column_stack([vec.squeeze() for vec in vecs])   # (N_total, N_vecs)
        c_evecs_array = np.column_stack([vec.squeeze() for vec in c_evecs])  # (N_total, N_cvecs)

        # Compute PCA in fully vectorized way
        PCA = np.abs(vecs_array.conj().T @ c_evecs_array)**2  # (len(vecs), len(cvecs))

        return PCA

    
    def get_map(PCA, used_indices=None):
        """
        Produce a mapping from bare states to dressed states using PCA overlaps,
        avoiding duplicates. If used_indices is given, never reuse these indices.
        """
        if used_indices is None:
            used_indices = set()

        mapp = np.argmax(PCA, axis=0)
        assigned = set()  # indices assigned in this call
        final_map = []

        for i, dressed_idx in enumerate(mapp):
            # Candidate must not be in either assigned (this call) or used_indices (previous calls)
            if dressed_idx not in assigned and dressed_idx not in used_indices:
                final_map.append(dressed_idx)
                assigned.add(dressed_idx)
            else:
                # Pick next best dressed state not used yet
                column = PCA[:, i]
                candidates = np.argsort(column)[::-1]  # descending
                for c in candidates:
                    if c not in assigned and c not in used_indices:
                        final_map.append(c)
                        assigned.add(c)
                        break
                else:
                    # Fallback: if somehow all indices are exhausted, just pick the next free
                    remaining = set(range(PCA.shape[0])) - assigned - used_indices
                    if remaining:
                        c = remaining.pop()
                        final_map.append(c)
                        assigned.add(c)
                    else:
                        raise RuntimeError("No available indices left to assign!")
        # Return both final_map and newly assigned indices
        return final_map, assigned
    
    """""""""
    Pt. 1: Object Initialization
    """""""""
    
    # Total space size
    N = qdim * rdim
    # Obtain dressed eigenvalues/eigenvectors
    dressed_evals, dressed_evecs = eigh(H_coupled_mat)
    qvecs_q = [qt.Qobj(q_evecs[:, -j], dims=[[qdim],[1]]) for j in range(qdim)]
    # Reorganizing vals/vecs

    # Create empty array to form ladder operator
    a = np.zeros((rdim, rdim), dtype=complex)
    # Populate ladder operator values
    for n in range(1, rdim):
        a[n-1, n] = np.sqrt(n)
    # Create a^{\dagger}
    adag = a.conj().T
    b_adag = np.kron(np.eye(qdim, dtype=complex), adag)
    # Create number operator
    n_res_op = adag @ a
    # Create resonator number operator same dimension as full H
    big_n = np.kron(np.eye(qdim, dtype=complex), n_res_op)
    # Create qubit population operator
    big_n_q = np.diag(np.arange(qdim)).astype(complex)
    # Convenient renaming
    V = dressed_evecs
    # rho_set: (N_states, qdim, qdim)
    # big_n_q: (qdim, qdim)
    # output: n_q_expect: (N_states,)
    n_q_expect = np.einsum('nij,ij->n', rho_set, big_n_q).real

    """""""""
    Pt. 2: Expectation Values
    """""""""
    
    # Evil Einstein sum for vectorized expectation values
    big_n_V = big_n @ V                                       # (N, M)
    n_r_expect = np.einsum('ij,ij->j', V.conj(), big_n_V).real  # (M,)

    M = V.reshape((qdim, rdim, V.shape[1]), order='C')  # shape (qdim, rdim, M)
    # Build rotation matrix from original bare qubit eigenvectors
    # Here, q_evecs is (qdim, qdim)
    U = np.column_stack([vec for vec in q_evecs])
    q_evecs = np.asarray([(U.conj().T @ vec) for vec in q_evecs])

    """""""""
    Pt. 3: Initial Branch Assignment
    """""""""
    
    # Get resonator vacuum state
    r_vac = r_evecs[0]
    # Construct qubit x vacuum states for first round of mapping
    bare_states = np.kron(q_evecs, r_vac)   # (qdim*rdim, qdim)
    PCA = ladder_levels(bare_states, c_evecs, b_adag)
    PCA_full = PCA.T
    # Obtain assignments and set of assigned indices
    fixed_map, assigned = get_map(PCA_full)
    # Initialize assignment objects
    PCA_list = [PCA_full]
    map_list = [fixed_map]
    used_idx = set(assigned)
    
    """""""""
    Pt. 4: Iterative Branch Assignment
    """""""""
    
    # Apply ladder operators and re-assign map up to max n_r
    for i in tqdm(range(rdim-1), ascii=True, desc="climbing ladders...", disable=flux_loop):
        vecs = c_evecs[np.asarray(map_list[-1])]
        PCA_temp = ladder_levels(vecs, c_evecs, b_adag, new_vecs=True)
        PCA_list.append(PCA_temp.T)
        fixed_map_temp, assigned = get_map(PCA_temp.T)
        map_list.append(fixed_map_temp)
        
    """""""""
    Pt. 5: Packaging
    """""""""
    
    # Build lists n_list1 (resonator excitation), 
    # n_list2 (qubit excitation) in dressed order
    n_list1 = n_r_expect.tolist()
    n_list2 = n_q_expect.tolist()
    
    # Bin branches based on map list
    branches = {}
    for ti in range(len(map_list[0])):   # Number of bare qubit states
        idx_list = [map_list[nj][ti] for nj in range(len(map_list))]
        n_r_arr = np.abs(np.array(n_list1)[idx_list])
        n_q_arr = np.abs(np.array(n_list2)[idx_list])
        branches[np.round(n_q_arr[0])] = ((n_r_arr, n_q_arr), None, None)

    # Params, data packaging
    params = {"qdim": (qdim, None, None), "rdim": (rdim, None, None)}
    data = {
        "PCA_list": (PCA_list, None, None),
        "map_list": (map_list, None, None),
        "avg_n_r": (n_list1, None, None),
        "avg_n_q": (n_list2, None, ["avg_n_r"]),
    }
    # Build lists for plotting
    data_list = []
    params_list = [params for _ in range(len(dressed_evals))]
    labels = []
    
    # For each bare state
    for t_idx in range(np.asarray(map_list).shape[1]):
        idx_list = []
        data_temp = data.copy()  # start from global data dict
        for nj in range(np.asarray(map_list).shape[0]):
            psi_index = map_list[nj][t_idx]
            idx_list.append(psi_index)

        # Extract expectation values along this branch
        n_r_branch = np.abs(np.asarray(n_list1)[idx_list])
        n_q_branch = np.abs(np.asarray(n_list2)[idx_list])

        # Sort by resonator population
        sort_idx = np.argsort(n_r_branch)
        n_r_branch = n_r_branch[sort_idx]
        n_q_branch = n_q_branch[sort_idx]

        # Label this branch by initial qubit population
        label = str(np.round(n_q_branch[0]))
        labels.append(label)

        # Populate data_temp
        data_temp['n_r'] = (n_r_branch, None, None)
        data_temp['n_q'] = (n_q_branch, None, ['n_r'])
        # Append to data_list
        data_list.append(data_temp)

        # Add to global data dictionary, label is qubit pop
        data[f'n_r{label}'] = (n_r_branch, None, None)
        data[f'n_q{label}'] = (n_q_branch, None, [f'n_r{label}'])
        
        dat_package = (params_list, data_list, 
                        branches, dressed_evals, dressed_evecs)

        dat_package_light = (params, data)

    return dat_package, dat_package_list

def comp_ss_dat(res_list, flux_arr, save=False):
    """
    Take flux sweep data and re-package it to look at 
    only the computational subspace.
    """
    # Initialize storage lists for ground/excited state
    q1_list = []
    r1_list = []
    q0_list = []
    r0_list = []
    
    # Unpack each result tuple and store ground/excited state data in lists
    for res in res_list:
        params, data, params_list, data_list, branches, dressed_evals, dressed_vecs = res
        q0_list.append(data['n_q0.0'][0])
        q1_list.append(data['n_q1.0'][0])
        r0_list.append(data['n_r0.0'][0])
        r1_list.append(data['n_r1.0'][0])
        
    # Convert lists to arrays
    q1_arr = np.asarray(q1_list)
    r1_arr = np.asarray(r1_list)
    q0_arr = np.asarray(q0_list)
    r0_arr = np.asarray(r0_list)

    # Taking norm of slope of <n_q> to see sudden changes in
    # qubit population.
    dq0 = np.abs(np.gradient(np.log(q0_arr), axis=1))
    dq1 = np.abs(np.gradient(np.log(q1_arr), axis=1))

    # Normalization
    dq0_norm = dq0 / (np.max(q0_arr) - np.min(q0_arr) + 1e-12)
    dq1_norm = dq1 / (np.max(q1_arr) - np.min(q1_arr) + 1e-12)

    dat0 = {'MIST_amp':(q0_arr, None, [r'$\Phi_{ext}$', r'$n_r$']),
            'MIST_grad':(dq0, None, [r'$\Phi_{ext}$', r'$n_r$']),
            'MIST_grad_norm':(dq0_norm, None, [r'$\Phi_{ext}$', r'$n_r$']),
            r'$n_r$':(r0_list[0], None, None),
            r'$\Phi_{ext}$':(flux_arr, r'$\Phi_0$', None)}
    
    dat1 = {'MIST_amp':(q1_arr, None, [r'$\Phi_{ext}$', r'$n_r$']),
            'MIST_grad':(dq1, None, [r'$\Phi_{ext}$', r'$n_r$']),
            'MIST_grad_norm':(dq1_norm, None, [r'$\Phi_{ext}$', r'$n_r$']),
            r'$n_r$':(r1_list[0], None, None),
            r'$\Phi_{ext}$':(flux_arr, r'$\Phi_0$', None)}
    
    return dat0, dat1
    