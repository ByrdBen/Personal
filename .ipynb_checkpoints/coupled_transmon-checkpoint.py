# transmon_plus_resonator.py
import numpy as np
import scqubits as scq
import qutip as qt 

class Coupled_Transmon(object):
    def __init__(self, EJ, EC, ng, ncut, tmon_trunc, res_freq, g, osc_trunc):
        self.EJ = EJ
        self.EC = EC
        self.ng = ng
        self.ncut = ncut
        self.tmon_trunc = tmon_trunc
        self.res_freq = res_freq
        self.g = g
        self.osc_trunc = osc_trunc
        
        tmon = scq.Transmon(EJ=EJ, EC=EC, ng=ng, ncut=ncut, truncated_dim=tmon_trunc, id_str="Tmon")
        resonator = scq.Oscillator(E_osc=res_freq, truncated_dim=osc_trunc, id_str="Res")

        hs = scq.HilbertSpace([tmon, resonator])
        
        hs.add_interaction(
            expr=f"{g} * n * (a + adag)",  # g is directly inserted
            op1=("n", tmon.n_operator(), tmon),
            op2=("a", resonator.annihilation_operator(), resonator),
            op3=("adag", resonator.creation_operator(), resonator),
            add_hc=False
        )
        self.hs = hs
        
        H_full = hs.hamiltonian()
        self.H = H_full        
        
        # Precompute eigenvalues and lookup table
        #self.evals = self.H.eigenvals()
        self.hs.generate_lookup()  # generate the lookup table
        
    def eigenvals(self, count=8):
        return self.H.eigenenergies()[:count]
        
    def w01(self):
        E = self.eigenvals(2)
        return (E[1] - E[0])
    
    def anharmonicity(self):
        E = self.eigenvals(3)
        return (E[2] - E[1]) - (E[1] - E[0])
       
    def w01_n(self, n_photons=0):
        """01 transition frequency for given photon number in resonator."""
        idx_0n = self.hs.dressed_index((0, n_photons))
        idx_1n = self.hs.dressed_index((1, n_photons))
        max_idx = np.max([idx_0n, idx_1n])
        evals = self.eigenvals(max_idx+1)
        return evals[idx_1n] - evals[idx_0n]

    def chi01(self, n_photons=0):
        """
        Dispersive shift of the qubit transition
        for a given photon number in the resonator.
        """
        return self.w01_n(n_photons) - self.w01_n(0)