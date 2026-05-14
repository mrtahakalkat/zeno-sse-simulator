"""
1D Quantum Transport & Measurement Simulator
Developer: Taha Kalkat

This script simulates how continuous measurement affects a quantum particle.
It includes:
1. High-res grid testing (N=4096, t_max=300)
2. Momentum sweeps (p0=3 vs p0=10)
3. Heavy tail & Sliding window beta analysis
4. Zeno freezing limit (Gamma=1000)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import concurrent.futures
import time

# --- 1. SETUP THE ENVIRONMENT ---
N = 4096
L_grid = 300.0
z = np.linspace(0, L_grid, N, endpoint=False)
dz = z[1] - z[0]
p = fftfreq(N, d=dz) * 2 * np.pi

t_max = 300.0
dt = 0.005
steps = int(t_max / dt)

# Kinetic energy operator
exp_T = np.exp(-1j * (p**2 / 2.0) * dt)

# CAP (Complex Absorbing Potential)
z_cap, L_cap, eta_cap = 220.0, 40.0, 10.0
f_z = np.where(z > z_cap, ((z - z_cap)/L_cap)**2, 0.0)
exp_V_half = np.exp(-eta_cap * f_z * dt / 2.0)

# --- 2. CORE ENGINE ---
def run_simulation(args):
    gamma, p0, w = args
    
    z_m = 40.0 
    M_z = np.exp(-((z - z_m)**2) / (2 * w**2))
    
    z0, sigma = 20.0, 2.0
    psi = (1.0 / (np.pi * sigma**2)**0.25) * np.exp(-((z - z0)**2) / (2 * sigma**2)) * np.exp(1j * p0 * z)
    psi /= np.linalg.norm(psi) * np.sqrt(dz)

    arrival_dist = np.zeros(steps)
    survival_list = np.zeros(steps)
    survival_prob = 1.0

    for i in range(steps):
        psi = exp_V_half * psi
        psi = ifft(exp_T * fft(psi))
        psi = exp_V_half * psi

        current_norm_sq = np.sum(np.abs(psi)**2) * dz
        loss = 1.0 - current_norm_sq
        arrival_dist[i] = loss * survival_prob / dt
        survival_prob *= current_norm_sq
        survival_list[i] = survival_prob
        
        if current_norm_sq < 1e-12: 
            break
        
        exp_M = np.sum(np.abs(psi)**2 * M_z) * dz
        dW = np.random.randn() * np.sqrt(dt)
        
        psi += (np.sqrt(gamma) * (M_z - exp_M) * dW - 0.5 * gamma * (M_z - exp_M)**2 * dt) * psi
        psi /= (np.linalg.norm(psi) * np.sqrt(dz))

    return arrival_dist, survival_list

# --- 3. SLIDING WINDOW (BETA) ANALYSIS ---
def calculate_sliding_beta(t_array, arrival_data, window_size=5000):
    betas = np.zeros(len(t_array) - window_size)
    for i in range(len(betas)):
        t_win = t_array[i:i+window_size]
        y_win = arrival_data[i:i+window_size]
        
        # FIX: Sadece olasılık değil, zamanın da 0'dan büyük olduğundan emin ol! (log(0) hatasını önler)
        valid = (y_win > 1e-20) & (t_win > 0.0)
        
        if np.sum(valid) > window_size // 2:
            coeffs = np.polyfit(np.log(t_win[valid]), np.log(y_win[valid]), 1)
            betas[i] = -coeffs[0]
        else:
            betas[i] = np.nan
    return betas

# --- 4. MAIN EXECUTION ---
if __name__ == '__main__':
    scenarios = [
        {'name': 'Zeno Freeze', 'gamma': 1000.0, 'p0': 5.0, 'w': 4.0},
        {'name': 'Low Momentum', 'gamma': 200.0, 'p0': 3.0, 'w': 4.0},
        {'name': 'High Momentum', 'gamma': 200.0, 'p0': 10.0, 'w': 4.0}
    ]
    
    num_traj = 50 # NOTE: Kept low for testing. Increase to 500+ for stable heavy tail analysis.
    t_arr = np.linspace(0, t_max, steps)
    
    print("Starting simulations... This will take some time.")
    
    results_dict = {}
    
    for sc in scenarios:
        print(f"Running: {sc['name']} (Gamma={sc['gamma']}, p0={sc['p0']})")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            args_list = [(sc['gamma'], sc['p0'], sc['w'])] * num_traj
            results = list(executor.map(run_simulation, args_list))
            
        avg_arr = np.mean([r[0] for r in results], axis=0)
        avg_surv = np.mean([r[1] for r in results], axis=0)
        results_dict[sc['name']] = (avg_arr, avg_surv)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Survival Plot
    for name, data in results_dict.items():
        axes[0].plot(t_arr, data[1], label=name)
    axes[0].set_title("Survival Probability")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Arrival Time (Log-Log) with Masking Fix
    for name, data in results_dict.items():
        mask = data[0] > 1e-20
        axes[1].loglog(t_arr[mask], data[0][mask], label=name)
    axes[1].set_title("Arrival Time (Log-Log)")
    axes[1].legend()
    axes[1].grid(True)

    # 3. Sliding Window Beta (Dynamic Target)
    beta_target = 'Low Momentum' 
    arr_target = results_dict[beta_target][0]
    betas = calculate_sliding_beta(t_arr, arr_target)
    axes[2].plot(t_arr[:-5000], betas, color='red')
    axes[2].set_title(f"Sliding Window Beta ({beta_target})")
    axes[2].set_ylim(0, 10)
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("final_combined_results.png")
    print("Done! Saved as final_combined_results.png")