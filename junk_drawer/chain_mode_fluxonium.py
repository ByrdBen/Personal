class Coupled_Fluxonium_w_Chain(object):
    """
    Generates coupled fluxonium/resonator hamiltonian.
    """
    def __init__(self, params):
        self.params = params

        # Set of allowed attributes
        attr = ['EJ', 'EC', 'EL', 'flux', 'ncut', 
                'f_trunc', 'f_r', 'g_n', 'g_phi', 'g_chain',
                'osc_trunc', 'chain_mode', 'chain_trunc', 
                'coupling_type']
        
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
        
        # Make resonator
        resonator = scq.Oscillator(self.f_r, 1/np.sqrt(2), truncated_dim=self.osc_trunc, 
                                   id_str="Res")
        
        self.resonator = resonator

        if (self.chain_mode):
            # Make chain mode
            chain_mode = scq.Oscillator(self.f_c, 1/np.sqrt(2), truncated_dim=self.chain_trunc, 
                               id_str="Chain")
            
            self.chain_mode = chain_mode

            # Combine circuit elements
            hs = scq.HilbertSpace([fluxonium, resonator, chain_mode])

        else:
            # Combine circuit elements
            hs = scq.HilbertSpace([fluxonium, resonator])
            
        # Turn on coupling
        if (self.coupling_type == "capacitive"): 
            hs.add_interaction(
                expr=f"{self.g_n} * n * (a + adag)",  # g is directly inserted
                op1=("n", fluxonium.n_operator(), fluxonium),
                op2=("a", resonator.annihilation_operator(), resonator),
                op3=("adag", resonator.creation_operator(), resonator),
                add_hc=False
            )
        if (self.coupling_type == "inductive"):
            hs.add_interaction(
                expr=f"{self.g_phi} * phi * (a - adag)",  # g is directly inserted
                op1=("phi", fluxonium.phi_operator(), fluxonium),
                op2=("a", resonator.annihilation_operator(), resonator),
                op3=("adag", resonator.creation_operator(), resonator),
                add_hc=False
            )

        if self.chain_mode:
            hs.add_interaction(
                expr=f"{self.g_chain} * sin_phi * (a + adag)",  # g is directly inserted
                op1=("sin_phi", fluxonium.sin_phi_operator(beta = -self.flux), fluxonium),
                op2=("a", chain_mode.annihilation_operator(), chain_mode),
                op3=("adag", chain_mode.creation_operator(), chain_mode),
                add_hc=False
            )

            
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
    
    def update_flux(self, flux, lookup=False, print_update=False):
        """
        Update value of flux in fluxonium, need to rebuild hamiltonian
        Lookup table generation optional, computationally expensive
        """
        self.flux = flux
        self.fluxonium.flux = flux
        self.hs = scq.HilbertSpace([self.fluxonium, self.resonator])
        
        # Turn on coupling
        if (self.coupling_type == "capacitive"): 
            self.hs.add_interaction(
                expr=f"{self.g_n} * n * (a + adag)",  # g is directly inserted
                op1=("n", self.fluxonium.n_operator(), self.fluxonium),
                op2=("a", self.resonator.annihilation_operator(), self.resonator),
                op3=("adag", self.resonator.creation_operator(), self.resonator),
                add_hc=False
            )
        if (self.coupling_type == "inductive"):
            self.hs.add_interaction(
                expr=f"{self.g_phi} * phi * (a - adag)",  # g is directly inserted
                op1=("phi", self.fluxonium.phi_operator(), self.fluxonium),
                op2=("a", self.resonator.annihilation_operator(), self.resonator),
                op3=("adag", self.resonator.creation_operator(), self.resonator),
                add_hc=False
            )
            
        self.H = self.hs.hamiltonian()
        
        if lookup:
            self.hs.generate_lookup()
            
        if print_update:
            print(r'Flux set to ' + f'{flux:.2f} ' + r'$\Phi_0$')

def get_g_chain(EJ, EC_a, EJ_a, cg_a, c_a, num_JJ, N):
    # See notes
    term1 = 2 * EC_a / EJ_a 
    term2 = 1 / (1 + (cg_a / c_a)*(num_JJ**2/(2**2 * np.pi**2)))
    return EJ * (term1 * term2) ** (1/4) * N ** (1/2)