import sys
import gc
from tqdm import tqdm
from dataclasses import dataclass
from branch_analysis_w_chain import *
from coupled_fluxonium import *
import os
import pickle
    
save_folder = f"/home/babyrd/branches/Personal/results/test/verification/" 
print("Save folder exists:", os.path.exists(save_folder), flush=True)
print("Python argv:", sys.argv)
print("Current dir:", os.getcwd())
print("Save folder:", save_folder)

chain_trunc = 6

# Setting some vals
f_trunc    = int(sys.argv[1])
ncut       = f_trunc
osc_trunc  = int(sys.argv[2])
flux       = .5
lookup     = False
EJ  = 7.3
EL  = .49
EC  = 1.73
g_n = .062
g_phi = 1j * g_n
f_r = 6.627
coupling_type = 'capacitive'
chain_mode = True

EJ_a = 60
EC_a = .74
cg_a = 1e-6
c_a  = 1e-6
num_JJ = 122
N = 0.0574
theta2 = .319
#g_chain = get_g_chain(EJ, EC_a, EJ_a, cg_a, c_a, num_JJ, 0.0574)
g_chain = EJ * theta2 * N ** (1/2)
f_c = np.sqrt(8 * EJ_a * EC_a)

fit_params = {}
# This may be excessive but it feels more flexible
# if things need to change
var_list = ['f_trunc', 'ncut', 'osc_trunc', 'flux', 'lookup', 'EJ', 'EL', 'EC', 
            'g_n', 'f_r', 'coupling_type', 'chain_mode', 'chain_trunc', 'g_chain',
            'f_c']

units = {'flux': r"$\Phi_0$"}
_locals = locals()

fit_params.update({
    name: (_locals[name], units.get(name), None)
    for name in var_list
})
    
fit_params['flux'] = (flux, r'$\Phi_0$', None)
dat_package, H_full = get_objs(fit_params, 'fluxonium')
params, data = branch_analysis(dat_package, update_flux=False)

# Path where you want to save all your files
os.makedirs(save_folder, exist_ok=True)  # create folder if it doesn't exist

# Example: save multiple files
filename = os.path.join(save_folder, f"res_flux_{flux:.4f}.pkl")
with open(filename, "wb") as f:
    pickle.dump(data, f)