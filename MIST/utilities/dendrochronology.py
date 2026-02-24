import os
import pickle
import numpy as np
import re

load_folder = r"/home/babyrd/branches/Personal/results/symmetry/"  # or "./results" for relative paths
save_folder = r"/home/babyrd/branches/Personal/results/composite/"  # or "./results" for relative paths
folders = os.listdir(load_folder)

for folder in folders:
    print(folder)
    path_full = os.path.join(load_folder, folder)
    res_list_loaded = []
    filenames = sorted(
        f for f in os.listdir(path_full)
        if f.endswith(".pkl")
    )
    n_files = len(filenames)
    for name in filenames:
        try:
            with open(os.path.join(path_full, name), "rb") as f:
                first_dict = pickle.load(f)
            print(f"Using {name} as template")
            break
        except (pickle.UnpicklingError, EOFError):
            skipped_files.append(name)
    
    # Grab useful keys
    keys = first_dict.keys()
    
    print(keys)
    # Define shape
    array_shape = first_dict[keys[0]][0].shape
    
    # Pre-allocate space in output dict 
    output = {
        k: np.zeros((n_files, *array_shape), dtype=first_dict[k][0].dtype)
        for k in keys
    }
    
    # Open and store each individual data set into output, then delete from memory
    for i, name in enumerate(filenames):
        filename = os.path.join(path_full, name)
        
        try:
            with open(filename, "rb") as f:
                d = pickle.load(f)
                
        except (pickle.UnpicklingError, EOFError) as e:
            skipped_files.append(name)
            continue
            
        if all(key in d for key in keys):
            for k in keys:
                output[k][i] = d[k][0]
        else:
            skipped_files.append(name)
        del d
            
    # Example: save multiple files
    filename = os.path.join(save_folder, f"{folder}_symmetry.pkl")
    with open(filename, "wb") as f:
        pickle.dump(output, f)
    
    # Write to text file
    filename = os.path.join(save_folder, f"{folder}_symmetry_skipped_points.txt")
    with open(filename, "w") as f:
        for name in skipped_files:
            line = name
            f.write(line + "\n")
    print(str(folder) + ' has been tended to!')