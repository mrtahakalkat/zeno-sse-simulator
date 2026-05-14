# Monitored Quantum Transport Simulator

**Developer:** Taha Kalkat  
**Language:** Python 3  

## What This Does

Simulates a 1D Gaussian wavepacket under continuous 
spatially localized quantum measurement.

Uses the Stochastic Schrödinger Equation (SSE) with 
Ito correction, Split-Step Fourier Method, and 
Complex Absorbing Potential (CAP) for arrival-time 
extraction.

## Key Findings

- Geometric selectivity: arrival-time broadening 
  occurs only when the detector overlaps with 
  the wavepacket spatially
- Momentum dependence: low-p₀ packets are trapped, 
  high-p₀ packets transmit ballistically
- Strong monitoring limit (γ=1000): suppressed 
  transport consistent with Zeno confinement

## Known Limitations & Open Questions

- 1D model only — 2D extension (real double-slit 
  geometry) is the next step
- Heavy tail exponent (β) requires t_max scaling 
  test to confirm artifact independence
- CAP reflection independence not yet verified 
  across all parameter regimes
- num_traj=50 is for testing — use 500+ for 
  statistically stable results

## Requirements

pip install numpy scipy matplotlib

## Usage

python simulation.py

## Physical Context

This work explores whether continuous localized 
monitoring reshapes quantum arrival-time statistics 
beyond simple decoherence broadening. Related to:
Wiseman & Milburn (Quantum Trajectories), 
Muga et al. (Arrival Time in QM),
Monitored Quantum Systems literature.