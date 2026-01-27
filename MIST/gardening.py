import os
import pickle
import numpy as np
import re
"""
This function has quickly become a memory bottleneck. Some quick digging 
suggests pre-allocating the memory, and populating the output dict 
one entry at a time is more effecient than loading and manipulating
all the data at once. Hoping this will fix it.
"""

save_folder = r"/home/babyrd/branches/Personal/results/test/"  # or "./results" for relative paths
folders = os.listdir(save_folder)

for folder in folders:
    basename = os.path.basename(folder)
    
    res_list_loaded = []
    filenames = sorted(
        f for f in os.listdir(folder)
        if f.endswith(".pkl")
    )
    
    string_list = ["q0_", "q1_"]
    n_files = len(filenames)
    skipped_files = []
    # Check first data file
    with open(os.path.join(save_folder, filenames[0]), "rb") as f:
        first_dict = pickle.load(f)
    
    # Grab useful keys
    keys = [
        k for k in first_dict.keys()
        if any(p in k for p in string_list) and re.search(r"_c\d+", k)]
    
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
        filename = os.path.join(save_folder, name)
        
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
        
    dat1 = {}
    dat0 = {}
    
    save_folder = r"/home/babyrd/branches/Personal/results/test/composite/"  # or "./results" for relative paths
    
    for key in output.keys():
        if 'q0_' in key:
            dat0[key] = output[key]
        if 'q1_' in key:
            dat1[key] = output[key]
            
    # Example: save multiple files
    filename = os.path.join(save_folder, f"{basename}_0.pkl")
    with open(filename, "wb") as f:
        pickle.dump(dat0, f)
    
    
    # Example: save multiple files
    filename = os.path.join(save_folder, f"{basename}_1.pkl")
    with open(filename, "wb") as f:
        pickle.dump(dat1, f)
    
    # Write to text file
    filename = os.path.join(save_folder, f"{basename}_skipped_points.txt")
    with open(filename, "w") as f:
        for name in skipped_files:
            line = name
            f.write(line + "\n")
