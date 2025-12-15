import os
import pickle
import numpy as np
import re

save_folder = r"/home/babyrd/branches/Personal/results/test/v2"  # or "./results" for relative paths

res_list_loaded = []
num_points =501
flux_array = np.linspace(-.2, .6, num_points)

for f_idx in flux_array:
    filename = os.path.join(save_folder, f"res_flux_{f_idx}.pkl")
    with open(filename, "rb") as f:
        res_list_loaded.append(pickle.load(f))

def combine_q_arrays(dict_list, q_pattern=("q0_", "q1_")):
    # ---- Step 1: Find all unique keys across all dicts ----
    all_keys = set()
    for d in dict_list:
        for k in d.keys():
            # Must contain q-pattern and also a "_cX" suffix
            if any(p in k for p in q_pattern):
                if re.search(r"_c\d+", k):
                    all_keys.add(k)
    
    # ---- Step 2: Determine array shape (from first available array) ----
    array_shape = None
    for d in dict_list:
        for k in d:
            if isinstance(d[k], tuple) and hasattr(d[k][0], "shape"):
                array_shape = d[k][0].shape
                break
        if array_shape is not None:
            break
    
    if array_shape is None:
        raise ValueError("No arrays found in dictionaries.")
    
    # ---- Step 3: Construct the stitched output ----
    output = {}
    for key in all_keys:
        rows = []
        for d in dict_list:
            if key in d:
                rows.append(d[key][0])       # take the 0th element of the tuple
            else:
                rows.append(np.zeros(array_shape))
        output[key] = np.stack(rows, axis=0)
    
    return output
    
dict_list = [res[1] for res in res_list]
output = combine_q_arrays(dict_list, q_pattern=("q0_", "q1_"))

dat1 = {}
dat0 = {}

for key in output.keys():
    if 'q0_' in key:
        dat0[key] = output[key]
    if 'q1_' in key:
        dat1[key] = output[key]

# Example: save multiple files
filename = os.path.join(save_folder, r"_0.pkl")
with open(filename, "wb") as f:
    pickle.dump(dat1, f)

print('here!')

# Example: save multiple files
filename = os.path.join(save_folder, r"_1.pkl")
with open(filename, "wb") as f:
    pickle.dump(dat0, f)
