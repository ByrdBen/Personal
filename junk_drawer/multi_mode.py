 # transmon_plus_resonator.py
import numpy as np
import scqubits as scq
import qutip as qt 
import re


class FluxoniumMM(object):
    """
    Generates coupled fluxonium/resonator hamiltonian.
    """
    def __init__(self, params):
        self.params = params

        # Set of allowed attributes
        attr = ['EJ', 'EC', 'EL', 'flux', 'ncut', 
                'f_trunc', 'mode_list', 'trunc_list', 
                'f_list', 'g_list', 'coupling_list']
        
        # Boolean to speed up computation regarding lookup table
        lookup = params['lookup'][0]
        for key, val in params.items():
            # Sick function that strips non-alphanumeric content from keys
            
            if 'E' in key:
                clean_key = re.sub(r'[^0-9a-zA-Z]', '', key)
            else:
                clean_key = re.sub(r'[^0-9a-zA-Z_]', '', key)
            # Don't wish to bog down class with too many params
            if clean_key in attr:
                # Set attributes of self to nice clean keys
                setattr(self, clean_key, val[0])
        
        # Make fluxonium
        fluxonium = scq.Fluxonium(EJ=self.EJ, EC=self.EC, EL=self.EL, flux=self.flux, 
                                  cutoff=self.ncut, truncated_dim=self.f_trunc, id_str="Fl")
        self.fluxonium = fluxonium
        subspace_list = [fluxonium]

        # Adding modes
        for i, mode in enumerate(self.mode_list):
            mode = scq.Oscillator(self.f_list[i], 1/np.sqrt(2), truncated_dim=self.trunc_list[i], 
                           id_str=f"mode{i}")
            subspace_list.append(mode)
            self.add_interaction(mode, coupling_list[i])

        # Combine circuit elements
        hs = scq.HilbertSpace(subspace_list)

        # Make some definitions for later
        self.hs = hs
        H_full = hs.hamiltonian()
        self.H = H_full        
        
        # Important note! Some functions will fail if lookup table not generated!  
        if lookup:
            # Precompute eigenvalues and lookup table
            self.hs.generate_lookup()
        
    def eigenvals(self, count=8):
        return self.H.eigenenergies()[:count]
        
    def w01(self):
        E = self.eigenvals(2)
        return (E[1] - E[0])
    
    def anharmonicity(self):
        E = self.eigenvals(3)
        return (E[2] - E[1]) - (E[1] - E[0])
       
    def w01_n(self, n_photons=0):
        """
        01 transition frequency for given photon number in resonator
        """
        if not lookup:
            print("Lookup Table Not Generated! Please Check Params!")
        else:
            idx_0n = self.hs.dressed_index((0, n_photons))
            idx_1n = self.hs.dressed_index((1, n_photons))
            max_idx = np.max([idx_0n, idx_1n])
            evals = self.hs.eigensys(max_idx+1)[0]
            return evals[idx_1n] - evals[idx_0n]
    
    def w02_n(self, n_photons=0):
        """
        02 transition frequency for given photon number in resonator
        """
        if not lookup:
            print("Lookup Table Not Generated! Please Check Params!")
        else:
            idx_0n = self.hs.dressed_index((0, n_photons))
            idx_2n = self.hs.dressed_index((2, n_photons))
            max_idx = np.max([idx_0n, idx_2n])
            evals = self.hs.eigensys(max_idx+1)[0]
            return evals[idx_2n] - evals[idx_0n]

    def chi01(self, n_photons=0):
        """
        0-1 dispersive shift at specific photon number
        """
        return self.w01_n(n_photons) - self.w01_n(0)
    
    def chi02(self, n_photons=0):
        """
        0-2 dispersive shift at specific photon number
        """
        return self.w02_n(n_photons) - self.w02_n(0)
    
    def get_qubit_drive(self):
        """
        Return fluxonium drive operator
        """
        a = qt.destroy(self.f_trunc)
        adag = a.dag()
        drive = qt.Qobj(adag - a)
        return qt.tensor(drive, qt.qeye(self.osc_trunc))
    
    def get_resonator_drive(self):
        """
        Return fluxonium drive operator
        """
        a = self.resonator.annihilation_operator()
        adag = self.resonator.creation_operator()
        drive_op = adag + a
        return qt.tensor(qt.qeye(self.f_trunc), qt.Qobj(drive_op))
    
    def get_n_operator(self):
        """
        Return fluxonium charge operator
        """
        n = self.fluxonium.n_operator()
        return qt.tensor(qt.Qobj(n), qt.qeye(self.osc_trunc))
    
    def update_flux(self, flux, lookup=False):
        self.flux = flux
        self.fluxonium.flux = flux
        self.params.flux = flux
        # rebuild Hilbert space
        self.__init__(self, self.params)   
        
        if lookup:
            self.hs.generate_lookup()
        
        if print_update:
            print(r'Flux set to ' + f'{flux:.2f} ' + r'$\Phi_0$')

def add_interaction(self, mode, coupling_type):
    
    if coupling_type == "chain":
        self.hs.add_interaction(
            expr=f"{self.g_chain} * sin_phi * (a + adag)",  # g is directly inserted
            op1=("sin_phi", self.fluxonium.sin_phi_operator(beta = -self.flux), self.fluxonium),
            op2=("a", mode.annihilation_operator(), mode),
            op3=("adag", mode.creation_operator(), mode),
            add_hc=False)
    
        self.hs.add_interaction(
            expr=f"{self.g_chain**2 / (2 * self.EJ)} * cos_phi * (a + adag)**2",  # g is directly inserted
            op1=("cos_phi", self.fluxonium.cos_phi_operator(beta = -self.flux), self.fluxonium),
            op2=("a", mode.annihilation_operator(), mode),
            op3=("adag", mode.creation_operator(), mode),
            add_hc=False)
    
        self.hs.add_interaction(
            expr=f"{-self.EJ_a / (4 * self.N**2)} * (phi**2) * (a + adag)**2",  # g is directly inserted
            op1=("phi", self.fluxonium.phi_operator(), self.fluxonium),
            op2=("a", mode.annihilation_operator(), mode),
            op3=("adag", mode.creation_operator(), mode),
            add_hc=False)
    
    if coupling_type == "capacitive":
        self.hs.add_interaction(
            expr=f"{self.g_n} * n * (a + adag)",  # g is directly inserted
            op1=("n", self.fluxonium.n_operator(), self.fluxonium),
            op2=("a", mode.annihilation_operator(), mode),
            op3=("adag", mode.creation_operator(), mode),
            add_hc=False)

    if coupling_type == "inductive":
        self.hs.add_interaction(
            expr=f"{self.g_phi} * phi * (a - adag)",  # g is directly inserted
            op1=("phi", self.fluxonium.phi_operator(), self.fluxonium),
            op2=("a", self.resonator.annihilation_operator(), self.resonator),
            op3=("adag", self.resonator.creation_operator(), self.resonator),
            add_hc=False)

def get_g_chain(EJ, EC_a, EJ_a, cg_a, c_a, num_JJ, N):
    # See notes
    term1 = 2 * EC_a / EJ_a 
    term2 = 1 / (1 + (cg_a / c_a)*(num_JJ**2/(2**2 * np.pi**2)))
    return EJ * (term1 * term2) ** (1/4) * N ** (1/2)


def LHL_freq(LL, CL, k, del_x):
    # Mode Structure in Superconducting Metamaterial Transmission Line Resonators by Haoyang Wang 2018
    return 1 / (2 * np.sqrt(LL * CL)) * 1 / (np.sin((k+1) * del_x / 2))