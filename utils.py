import numpy as np

# -----------------------------
# Causal double-exponential filters [Frémaux et al. 2013]
#
# The double-exponential kernel f(t) = (exp(-t/τ_slow) - exp(-t/τ_fast)) / (τ_slow - τ_fast)
# models both PSP shapes ε (τ_s, τ_m) (Eq 19) and the value-readout kernel k (u_k, t_k) (Eq 14).
# The two running traces (fast and slow) compactly represent the full convolution history.
# -----------------------------

def step_double_exp_exact(trace_fast, trace_slow, spikes, dt, tau_fast, tau_slow):
    """
    Advance two exponential traces by one dt step, driven by an impulse input.

    Exact solution of dT/dt = -T/τ + spikes(t), for piecewise-constant spikes:
      T(t+dt) = T(t) · exp(-dt/τ) + spikes
    The double-exp convolution value is recovered via double_exp_value().
    """
    decay_fast = np.exp(-dt / tau_fast)
    decay_slow = np.exp(-dt / tau_slow)
    trace_fast = trace_fast * decay_fast + spikes
    trace_slow = trace_slow * decay_slow + spikes
    return trace_fast, trace_slow

def double_exp_value(trace_fast, trace_slow, tau_fast, tau_slow):
    """
    Recover the double-exp convolution value from the two running traces.
    (Y * f)(t) = (trace_slow - trace_fast) / (τ_slow - τ_fast)
    """
    return (trace_slow - trace_fast) / (tau_slow - tau_fast)


def k_and_kdot_from_traces(tr_uk, tr_tk, uk, tk):
    """
    Recover k(t) and k̇(t) from traces accumulated with time constants (uk, tk).

    k(t) = (exp(-t/tk) - exp(-t/uk)) / (tk - uk)
    k̇(t) = (-exp(-t/tk)/tk + exp(-t/uk)/uk) / (tk - uk)

    Used to compute V(t) and V̇(t) for the continuous-time TD error [Eq. 16].
    """
    k = (tr_tk - tr_uk) / (tk - uk)
    kdot = ((-tr_tk / tk) - (-tr_uk / uk)) / (tk - uk)

    k = np.nan_to_num(k, nan=0.0, posinf=100.0, neginf=-100.0)
    kdot = np.nan_to_num(kdot, nan=0.0, posinf=1000.0, neginf=-1000.0)

    return k, kdot
