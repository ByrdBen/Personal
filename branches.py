import sys
import gc
from tqdm import tqdm
from dataclasses import dataclass
from branch_analysis_w_chain import *
import os
import pickle

save_folder = r"/home/babyrd/branches/Personal/results/test/v1/"  # or "./results" for relative paths
print("Save folder exists:", os.path.exists(save_folder), flush=True)
print("Python argv:", sys.argv)
print("Current dir:", os.getcwd())
print("Save folder:", save_folder)

# Setting some vals
f_trunc    = int(sys.argv[1])
ncut       = f_trunc
osc_trunc  = int(sys.argv[2])
flux       = float(sys.argv[3])
lookup     = False
EJ  = 2.040
EL  = .091
EC  = 1.083
g_n = .062
g_phi = 1j * g_n
f_r = 6.627
coupling_type = 'capacitive'
chain_mode = False
fit_params = {}
# Tuple for moving stuff around

# This may be excessive but it feels more flexible
# if things need to change
var_list = ['f_trunc', 'ncut', 'osc_trunc', 'flux', 'lookup', 'EJ', 'EL', 'EC', 'g_n', 'g_phi', 'f_r', 'coupling_type', 'chain_mode']
units = {'flux': r"$\Phi_0$"}
_locals = locals()

fit_params.update({
    name: (_locals[name], units.get(name), None)
    for name in var_list
})
    
fit_params['flux'] = (flux, r'$\Phi_0$', None)
dat_package, H_full = get_objs(fit_params, 'fluxonium')
res = branch_analysis(dat_package, update_flux=True)
del dat_package
gc.collect()

# Path where you want to save all your files
os.makedirs(save_folder, exist_ok=True)  # create folder if it doesn't exist

# Example: save multiple files
filename = os.path.join(save_folder, f"res_flux_{flux}.pkl")
with open(filename, "wb") as f:
    pickle.dump(res, f)