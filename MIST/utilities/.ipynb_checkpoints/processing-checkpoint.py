import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colors, cm 
import pickle
import os as os

def peak_detection(data, key, ref_val, freq_range, threshold=.5):
    MIST_dat = np.flipud(data[key].T)
    shifted_dat = np.abs(MIST_dat - ref_val)
    deriv = np.gradient(shifted_dat, axis=0)
    num_peaks = 0
    for column in deriv:
        jump_indices = np.where(column > threshold)[0]
        gaps = np.diff(jump_indices)
        tot = 1 + np.sum(gaps > 1)  
        num_peaks += tot
        
    peaks_per = num_peaks / freq_range
    
    return num_peaks, peaks_per

def get_MIST_dens(data, freq_range, n_r, threshold=.5):
    key_list = [key for key in data.keys() if 'n_q' in key]
    for key in key_list:
        if '=q0' in key:
            ref_val = 0
        if '=q1' in key:
            ref_val = 1
        num_peaks, peaks_per, n_crit = peak_detection_2(data, key, ref_val, freq_range, threshold=threshold)
        
        tot_key  = key + '_tot'
        dens_key = key + '_dens'
        crit_key = key + '_n_crit'
        dat = data.copy()
        dat[tot_key] = (num_peaks, None, ['n_r'])
        dat[dens_key] = (peaks_per, None, ['n_r'])
        dat[crit_key] = (n_crit, None, ['flux'])
        dat[r'$\langle n_r \rangle$'] = (n_r, None, None) 
        
    return dat

def peak_detection_2(data, key, ref_val, flux_range, threshold=.5, cluster_window=5):
      MIST_dat = np.flipud(data[key].T)  # shape (n_flux, n_r) - don't transpose
      shifted_dat = np.abs(MIST_dat - ref_val)
      deriv = np.abs(np.gradient(shifted_dat, axis=1))  # along n_r

      n_flux, n_r_len = MIST_dat.shape
      peaks_vs_nr = np.zeros(n_r_len)
      n_crit = np.zeros(n_flux)
    
      for flux_idx in range(n_flux):
          row = deriv[flux_idx, :]
          jump_indices = np.where(row > threshold)[0]
          if len(jump_indices) == 0:
              continue
    
          # Cluster and assign each transition to a single n_r
          gaps = np.diff(jump_indices)
          split_points = np.where(gaps > cluster_window)[0] + 1
          clusters = np.split(jump_indices, split_points)
    
          for cluster in clusters:
              center = cluster[len(cluster) // 2]
              peaks_vs_nr[center] += 1
          n_crit[flux_idx] = clusters[0][0]
    
        
      # peaks_vs_nr[i] = number of flux points with a transition at n_r = i
      density_vs_nr = peaks_vs_nr / flux_range  # transitions per GHz at each n_r

      return np.cumsum(peaks_vs_nr), density_vs_nr, n_crit

def plot_trajectories(data, q_key, c_key, init_val, save=False):
    ref_dat = data[q_key]
    num_pts = len(ref_dat)
    flux_vals = np.linspace(0, .5, num_pts)
    fig, ax = plt.subplots(dpi = 500)
    ax.set_yscale('log')
    ax.set_xscale('log')

    cmap = cm.viridis
    norm = colors.Normalize(vmin=flux_vals.min(), vmax=flux_vals.max())
    
    for j in range(num_pts):
        x = data[q_key][j]
        y = data[c_key][j]
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        speed = np.sqrt(dx**2 + dy**2)
    
        cut = np.percentile(speed, 80)
        mask = speed < cut
        
        N = 3
        i = np.where(mask)[0][::N]
        
        Q = ax.quiver(
            x[i], y[i],
            dx[i], dy[i],
            np.full(len(i), flux_vals[j]),
            cmap='plasma',
            norm=norm,
            angles='xy',
            scale_units='xy',
            scale=0.15,
            alpha=.5,
            
        )
    
    fig.colorbar(Q, ax=ax, label=r"External Flux $(\Phi_0)$")
    plt.xlabel(r"$\langle n_q(\langle n_r\rangle) \rangle$")
    plt.ylabel(r"$\langle n_c(\langle n_r\rangle) \rangle$")
    plt.title(r"MIST Trajectory, $|\Psi_{init}\rangle = |0, 0, 0\rangle$")
    plt.grid(alpha=.2)
    
    plt.ylim(1e-3, 10)
    plt.xlim(1e-3 + init_val, 20)
    if save:
        plt.savefig('Ground_Trajectory.png')
    plt.show()

def get_expects(folder, cdim):
    sub_folders = os.listdir(folder)
    N_E = len(sub_folders)
    N_flux = len(os.listdir(os.path.join(folder, sub_folders[0])))
    
    first_sub = os.path.join(folder, sub_folders[0])
    first_file = os.path.join(first_sub, os.listdir(first_sub)[0])
    with open(first_file, "rb") as f:
        d = pickle.load(f)
        c_bins = np.arange(0, cdim+1, 1.)
        unique_c_bins = sorted(set(int(v) for v in c_bins))
    
    composite = {c: np.full((N_E, N_flux), np.nan) for c in unique_c_bins}
    print(composite.keys())
    
    for i, path in enumerate(sub_folders):
        full_path = os.path.join(folder, path)
        sub_paths = os.listdir(full_path)
        
        for j, name in enumerate(os.listdir(full_path)):
            filename = os.path.join(full_path, name)
            
            with open(filename, "rb") as f:
                d = pickle.load(f)
                c_dat = d['chain_occupation_number'][0]
                c_bins = np.round(np.abs(c_dat))
                
                p_dat = np.abs(d['parity'][0])
                n_dat = d['excitation_number'][0]

                unique_c_bins = sorted(set(int(v) for v in c_bins))
                c_dim = len(unique_c_bins)
                                
                for c_val in unique_c_bins:
                    mask = (c_bins == c_val)
                    if mask.any():
                        composite[c_val][i, j] = np.sum(p_dat[mask] < 0.8) / mask.sum()

    return composite
                