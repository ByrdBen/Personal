import numpy as np 

def peak_detection(data, key, ref_val, freq_range, threshold=.5):
    MIST_dat = data[key].T
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

def get_MIST_dens(data, freq_range, n_r):
    key_list = [key for key in data.keys() if 'n_q' in key]
    for key in key_list:
        if '=q0' in key:
            ref_val = 0
        if '=q1' in key:
            ref_val = 1
        num_peaks, peaks_per, n_crit = peak_detection_2(data, key, ref_val, freq_range)
        
        tot_key  = key + '_tot'
        dens_key = key + '_dens'
        crit_key = key + '_n_crit'
        data[tot_key] = (num_peaks, None, ['n_r'])
        data[dens_key] = (peaks_per, None, ['n_r'])
        data[crit_key] = (n_crit, None, ['flux'])
        data[r'$\langle n_r \rangle$'] = (n_r, None, None) 
        
    return data

def peak_detection_2(data, key, ref_val, flux_range, threshold=.5, cluster_window=5):
      MIST_dat = np.flipud(data[key])  # shape (n_flux, n_r) - don't transpose
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
          n_crit[flux_idx] = (clusters[0][-1] - clusters[0][0])/2
    
        
      # peaks_vs_nr[i] = number of flux points with a transition at n_r = i
      density_vs_nr = peaks_vs_nr / flux_range  # transitions per GHz at each n_r

      return np.cumsum(peaks_vs_nr), density_vs_nr, n_crit

def plot_trajectories(data, q_key, c_key):
    ref_dat = data[q_key]
    num_pts = len(ref_dat)
    flux_vals = np.linspace(-.2, .6, num_pts)
    from matplotlib import colors, cm 
    fig, ax = plt.subplots()
    cmap = cm.viridis
    norm = colors.Normalize(vmin=flux_vals.min(), vmax=flux_vals.max())
    
    for j in range(num_pts):
        x = data[q_key][j]
        y = data[c_key][j]
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        speed = np.sqrt(dx**2 + dy**2)
    
        cut = np.percentile(speed, 90)
        mask = speed < cut
        
        N = 1
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
            alpha=.8
        )
    
    fig.colorbar(Q, ax=ax, label=r"External Flux $(\Phi_0)$")
    plt.xlabel(r"$\langle n_q(\langle n_r\rangle) \rangle$")
    plt.ylabel(r"$\langle n_c(\langle n_r\rangle) \rangle$")
    plt.title(r"MIST Trajectory, $|\Psi_{init}\rangle = |0, 0, 0\rangle$")
    plt.grid(alpha=.2)
    plt.ylim(0, 3.5)
    plt.xlim(0, 12)
    if save:
        plt.savefig('Ground_Trajectory.png')
    plt.show()
