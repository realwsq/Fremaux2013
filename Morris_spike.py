import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dataclasses import dataclass
from tqdm import tqdm

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
})

from maze import Maze
from utils import k_and_kdot_from_traces, step_double_exp_exact, double_exp_value

@dataclass
class Params:
    # time
    dt: float = 1e-3       # 1 ms
    Tclamp: float = 0.5    # 500 ms clamped startup before TD learning begins
    Ttimeout: float = 50.0 # trial cutoff (s)

    # place cells (Eq. 22)
    dPC: float = 2.0       # grid spacing
    sPC: float = 1.0       # tuning width
    rPC: float = 1000.0    # peak firing rate (Hz)

    # SRM0 neuron model 
    tm: float = 0.020   # membrane time constant τ_m = 20 ms
    ts: float = 0.005   # synaptic time constant τ_s = 5 ms
    e0: float = 0.004   # EPSP amplitude for PC→critic/actor
    e0_rec: float = 0.020   # EPSP amplitude for actor recurrent connections (preserves ring attractor)
    x_ref: float = -5   # refractory kernel amplitude η_0 = -5 mV
    r0: float = 60.0    # baseline firing rate r_0 = 60 Hz
    h: float = 16       # threshold h = 16 mV
    Du: float = 2       # noise slope Δu = 2 mV

    # critic / value readout (Eq. 13)
    Ncritic: int = 100
    tr: float = 4.0     # reward discount time constant τ_r = 4 s [eq 4]
    V0: float = -20.0   # value baseline offset
    v: float = 2.0      # value scale v = 2 reward·s 

    # double-exponential k kernel (Eq. 14)
    uk: float = 0.1     # fast time constant u_k = 100 ms
    tk: float = 0.5     # slow time constant t_k = 500 ms

    # actor 
    Nactor: int = 180   # number of directional neurons
    a0: float = 1.8     # velocity per rate unit [between eq 32 and 33]
    uc: float = 0.020   # fast PSP kernel time constant u_c = 20 ms
    tc: float = 0.050   # slow PSP kernel time constant t_c = 50 ms

    # actor ring-attractor lateral connectivity (Eq. 32-33)
    w_minus: float = -60.0
    w_plus: float = 30.0
    xi: float = 8.0

    # weight initialization and TD-LTP learning rates 
    mw: float = 5.0     # initial weight mean
    sw: float = 0.0     # initial weight std
    wmin: float = 0.0
    wmax: float = 30.0
    g_critic: float = 0.02   # critic learning rate g_c
    g_actor: float = 1e-4    # actor learning rate g_a

class ACMaze:
    """
    Continuous-time actor-critic agent with spiking neurons for the Morris water maze.

    Implements the model from Frémaux et al. (2013) "Reinforcement Learning Using a
    Continuous Time Actor-Critic Framework with Spiking Neurons", PLOS Computational Biology.

    Architecture:
      - Place cells: Gaussian tuning curves → Poisson spikes encoding position
      - Critic (100 SRM0 neurons): learns value function V(x) via TD-LTP
      - Actor (180 SRM0 neurons, ring attractor): outputs velocity direction via TD-LTP
      - Learning rule: TD-LTP (Eq. 17) 
    """
    def __init__(self, seed=0, params=Params(), maze=Maze()):
        self.p = params
        self.maze = maze
        self.rng = np.random.default_rng(seed)

        # γ = 1 − dt/τ_r  [Eq. 19]: discrete-time approximation of continuous discount
        self.p.gamma = 1.0 - self.p.dt / self.p.tr

        self.init_place_cells()

        self.init_critic()
        self.init_actor()

    def init_place_cells(self):
        """
        Build a uniform grid of place cell centers covering the maze (Eq. 22).
        Each center x_j has a Gaussian tuning curve.
        """
        grid_x = np.arange(-self.maze.half_size-self.p.dPC, self.maze.half_size + self.p.dPC+1e-3, self.p.dPC)
        grid_y = np.arange(-self.maze.half_size-self.p.dPC, self.maze.half_size + self.p.dPC+1e-3, self.p.dPC)
        gx, gy = np.meshgrid(grid_x, grid_y)
        self.p.pc_centers = np.vstack([gx.ravel(), gy.ravel()]).T
        self.p.Npc = self.p.pc_centers.shape[0]

    def init_critic(self):
        # Critic weights: initialized from N(mw, sw)
        self.Wc = self.rng.normal(loc=self.p.mw, scale=self.p.sw, size=(self.p.Ncritic, self.p.Npc))
        self.Wc = np.clip(self.Wc, self.p.wmin, self.p.wmax)

    def _build_actor_recurrent(self):
        """
        Build the fixed ring-attractor recurrent weight matrix for the actor [Eq. 32].
        The ring attractor amplifies the strongest direction and suppresses others,
        stabilizing the continuous heading representation.
        """
        N = self.p.Nactor
        k = np.arange(1, N + 1)
        theta = 2 * np.pi * k / N
        dtheta = theta[:, None] - theta[None, :]
        f = np.exp(self.p.xi * np.cos(dtheta))
        np.fill_diagonal(f, 0.0)
        Zf = f.sum(axis=1, keepdims=True)
        W = (self.p.w_minus / N) + self.p.w_plus * (f / Zf)
        return W
    
    def init_actor(self):
        # Actor weights: initialized from N(mw, sw)
        self.Wa = self.rng.normal(loc=self.p.mw, scale=self.p.sw, size=(self.p.Nactor, self.p.Npc))
        self.Wa = np.clip(self.Wa, self.p.wmin, self.p.wmax)

        self.Wrec = self._build_actor_recurrent()

        # Preferred directions for actor neurons
        k = np.arange(1, self.p.Nactor + 1)
        theta = 2 * np.pi * k / self.p.Nactor
        self.p.actor_dirs = self.p.a0 * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    
    def place_rates(self, x):
        """Gaussian tuning curves for all Npc place cells [Eq. 22]."""
        d = self.p.pc_centers - x[None, :]
        return self.p.rPC * np.exp(-(d[:, 0]**2 + d[:, 1]**2) / (self.p.sPC**2))

    def poisson_spike(self, rates):
        """
        Sample place cell spikes: X_j ~ Poisson(r_j(x)·dt).
        """
        return self.rng.poisson(rates * self.p.dt)

    def srm0_step(self, W_in, epsp_in, last_spike, extra_current=None, p_ub=0.02):
        """
        One dt step of the SRM0 (Spike Response Model 0) neuron [Eq. 18-20].
        """
        # Synaptic drive: u = W · (place-cell PSP convolution)
        u = W_in @ epsp_in

        # Optional extra current (actor recurrent connections)
        if extra_current is not None:
            u = u + extra_current

        # Refractory kernel 
        t = self.t
        dt_last = t - last_spike
        ref = self.p.x_ref * np.exp(-dt_last / self.p.tm)
        ref[~np.isfinite(dt_last)] = 0.0
        u = u + ref

        # Stochastic spiking
        rate = self.p.r0 * np.exp((u - self.p.h) / self.p.Du)
        pspk = np.clip(rate * self.p.dt, 0.0, p_ub)
        spikes = (self.rng.random(rate.shape) < pspk).astype(np.float32)

        last_spike[spikes > 0.0] = t
        return spikes, last_spike, u, pspk

    
    def td_ltp_update(self, Xpc_prev, pc_ts_since_a, pc_tm_since_a, Ya, elig_a_uk, elig_a_tk,
                      delta, g):
        """
        TD-LTP learning rule for one layer of PC→neuron weights [Eq. 17].
        Works for both critic (Wc, Ya=Yc) and actor (Wa, Ya=actor spikes).
        """
        # Accumulate pre-EPSP traces since last post-spike (reset at each post-spike below)
        pc_ts_since_a, pc_tm_since_a = step_double_exp_exact(
            pc_ts_since_a, pc_tm_since_a, Xpc_prev[None, :], self.p.dt, self.p.ts, self.p.tm
        )
        pc_epsp_since_a = self.p.e0 * double_exp_value(
            pc_ts_since_a, pc_tm_since_a, self.p.ts, self.p.tm
        )

        # Hebbian coincidence: H_ij = post_spike_i × pre_EPSP_j(since last spike_i)
        Ha = (Ya[:, None] * pc_epsp_since_a)

        # Filter H through k kernel to get eligibility trace e_ij(t) 
        elig_a_uk, elig_a_tk = step_double_exp_exact(
            elig_a_uk, elig_a_tk, Ha, self.p.dt, self.p.uk, self.p.tk
        )
        elig_a = double_exp_value(elig_a_uk, elig_a_tk, self.p.uk, self.p.tk)

        # ΔW = g · δ · elig  [Eq. 17]
        dWa = (g * delta) * elig_a

        # Reset pre-EPSP accumulator for neurons that just spiked
        spa = Ya > 0.0
        if np.any(spa):
            pc_ts_since_a[spa, :] = 0.0
            pc_tm_since_a[spa, :] = 0.0

        return dWa, pc_ts_since_a, pc_tm_since_a, elig_a_uk, elig_a_tk

    def run_trial(self, start_pose):
        self.t = 0.0
        x = np.array(start_pose)
        rates_pc = self.place_rates(x)
        Xpc = self.poisson_spike(rates_pc)

        # reset eligibility traces
        pc_ts = np.zeros(self.p.Npc)
        pc_tm = np.zeros(self.p.Npc)
        epsp_pc = np.zeros(self.p.Npc)
        Yc = np.zeros(self.p.Ncritic)
        c_k_tk = np.zeros(self.p.Ncritic)
        c_k_uk = np.zeros(self.p.Ncritic)
        Ya = np.zeros(self.p.Nactor)
        a_ts = np.zeros(self.p.Nactor)
        a_tm = np.zeros(self.p.Nactor)
        a_c_uc = np.zeros(self.p.Nactor)
        a_c_tc = np.zeros(self.p.Nactor)
        last_spike_c = -np.inf * np.ones(self.p.Ncritic)
        last_spike_a = -np.inf * np.ones(self.p.Nactor)
        pc_ts_since_c = np.zeros((self.p.Ncritic, self.p.Npc))
        pc_tm_since_c = np.zeros((self.p.Ncritic, self.p.Npc))
        pc_ts_since_a = np.zeros((self.p.Nactor, self.p.Npc))
        pc_tm_since_a = np.zeros((self.p.Nactor, self.p.Npc))
        elig_c_uk = np.zeros((self.p.Ncritic, self.p.Npc))
        elig_c_tk = np.zeros((self.p.Ncritic, self.p.Npc))
        elig_a_uk = np.zeros((self.p.Nactor, self.p.Npc))
        elig_a_tk = np.zeros((self.p.Nactor, self.p.Npc))

        # save history for analysis
        self.history = {'t': [],
                        'x': [],
                        'rates_pc': [],
                        'V': [],
                        'r_actor': [],
                        'a': [],
                        'r': [],
                        'delta': [],
                        }
        reached = False
        reached_time = None

        for step_count, self.t in tqdm(enumerate(np.arange(0.0, self.p.Ttimeout, self.p.dt))):
            # ---- Actor: ring attractor + population vector readout ----
            # Actor updates spikes
            a_ts, a_tm = step_double_exp_exact(
                a_ts, a_tm, Ya, self.p.dt, self.p.ts, self.p.tm
            )
            epsp_a = self.p.e0_rec * double_exp_value(a_ts, a_tm, self.p.ts, self.p.tm)
            rec_current = self.Wrec @ epsp_a
            Ya, last_spike_a, ua, pa = self.srm0_step(self.Wa, epsp_pc, last_spike_a, extra_current=rec_current, p_ub=0.02)[:4]
            # Smooth actor spike train 
            a_c_uc, a_c_tc = step_double_exp_exact(
                a_c_uc, a_c_tc, Ya, self.p.dt, self.p.uc, self.p.tc
            )
            r_a = double_exp_value(a_c_uc, a_c_tc, self.p.uc, self.p.tc)
            # Population vector
            a = (r_a[:, None] * self.p.actor_dirs).sum(axis=0) / self.p.Nactor

            # ---- Environment step ----
            x, event_R = self.maze.step(x, a, self.p.dt)
            if self.maze.goal_reached(x):
                event_R += self.maze.Rgoal
                reached = True
                reached_time = self.t

            r_t = event_R  

            # ---- Critic: value function V(x) ----
            # V(t) = (n/N) · Σ_i (Y_i * k)(t) + V0,  where k = double-exp kernel
            Yc, last_spike_c = self.srm0_step(self.Wc, epsp_pc, last_spike_c)[:2]
            c_k_uk, c_k_tk = step_double_exp_exact(
                c_k_uk, c_k_tk, Yc, self.p.dt, self.p.uk, self.p.tk
            )

            # ---- TD error δ(t) = γ·V(t+dt) − V(t) + r(t) ----
            # V(t+dt) ≈ V(t) + V̇(t)·dt  (first-order Taylor; V̇ from derivative of k kernel)
            # γ = 1 − dt/τ_r 
            if self.t < self.p.Tclamp:
                V_prev = 0.
                V = 0.
            else:
                k_val, k_dot = k_and_kdot_from_traces(c_k_uk, c_k_tk, self.p.uk, self.p.tk)
                V_prev = (self.p.v / self.p.Ncritic) * np.sum(k_val) + self.p.V0
                Vdot = (self.p.v / self.p.Ncritic) * np.sum(k_dot)
                V = V_prev + Vdot * self.p.dt

            if reached:
                V = np.zeros_like(V)  

            delta = self.p.gamma * V - V_prev + r_t

            # ---- TD-LTP weight updates ----
            ret_c = self.td_ltp_update(Xpc, pc_ts_since_c, pc_tm_since_c, Yc, elig_c_uk, elig_c_tk, delta, self.p.g_critic)
            dWc, pc_ts_since_c, pc_tm_since_c, elig_c_uk, elig_c_tk = ret_c
            self.Wc += dWc
            self.Wc = np.clip(self.Wc, self.p.wmin, self.p.wmax)
            ret_a = self.td_ltp_update(Xpc, pc_ts_since_a, pc_tm_since_a, Ya, elig_a_uk, elig_a_tk, delta, self.p.g_actor)
            dWa, pc_ts_since_a, pc_tm_since_a, elig_a_uk, elig_a_tk = ret_a
            self.Wa += dWa
            self.Wa = np.clip(self.Wa, self.p.wmin, self.p.wmax)   
            
            
            # ---- Place cells: new position → Poisson spikes → PSP convolution ----
            rates_pc = self.place_rates(x)          
            Xpc = self.poisson_spike(rates_pc)      
            pc_ts, pc_tm = step_double_exp_exact(
                pc_ts, pc_tm, Xpc, self.p.dt, self.p.ts, self.p.tm
            )
            rates_est_pc = double_exp_value(pc_ts, pc_tm, self.p.ts, self.p.tm)
            epsp_pc = self.p.e0 * rates_est_pc

            # save history
            self.history['t'].append(self.t)
            self.history['x'].append(x)
            self.history['rates_pc'].append(rates_pc)
            self.history['V'].append(V)
            self.history['a'].append(a)
            self.history['r_actor'].append(r_a)
            self.history['r'].append(r_t)
            self.history['delta'].append(delta)

            if reached:
                break
        
        self.history['reached'] = reached
        self.history['reached_time'] = reached_time
        return self.history
    
    def plot_traj(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 9))

        traj = np.array(self.history['x'])
        points = np.array([traj[:, 0], traj[:, 1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Line collection colored by time
        lc = LineCollection(segments, cmap='viridis')
        lc.set_array(np.arange(len(segments)))
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.plot(traj[0, 0], traj[0, 1], 'co', markersize=10, label='Start', markeredgecolor='black')
        if self.history['reached']:
            ax.plot(traj[-1, 0], traj[-1, 1], 'y*', markersize=15, label='End (reached)', markeredgecolor='black')
        else:
            ax.plot(traj[-1, 0], traj[-1, 1], 'rx', markersize=15, label='End (timeout)', markeredgecolor='black')

        return ax
    
    def plot_valuemap(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 9))

        # Compute value map across maze
        grid_size = int(np.sqrt(self.p.Npc))-2
        grid_size *= 2
        hs = self.maze.half_size
        x_range = np.linspace(-hs, hs, grid_size)
        y_range = np.linspace(-hs, hs, grid_size)
        X, Y = np.meshgrid(x_range, y_range)

        V_map = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                pos = np.array([X[i, j], Y[i, j]])
                rates_pc = self.place_rates(pos)
                V_map[i, j] = np.mean(self.Wc @ rates_pc)

        # Plot value map as heatmap
        im = ax.contourf(X, Y, V_map, levels=20, cmap='viridis', alpha=0.7)
        plt.colorbar(im, ax=ax, label='Value Function V(x)')
        return ax
    
    def plot_actormap(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 9))

        # Compute actor weight map across maze
        grid_size = int(np.sqrt(self.p.Npc))-2
        hs = self.maze.half_size
        x_range = np.linspace(-hs, hs, grid_size)
        y_range = np.linspace(-hs, hs, grid_size)
        X, Y = np.meshgrid(x_range, y_range)

        U = np.zeros((grid_size, grid_size))
        V = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                pos = np.array([X[i, j], Y[i, j]])
                rates_pc = self.place_rates(pos)
                r_actor = self.Wa @ rates_pc
                a = (r_actor[:, None] * self.p.actor_dirs).sum(axis=0) / self.p.Nactor
                U[i, j], V[i, j] = a

        norm = np.sqrt(U**2 + V**2)
        U = U / (norm + 1e-6)
        V = V / (norm + 1e-6)
        # Plot the policy as a vector field overlay
        quiver = ax.quiver(X, Y, U, V,
                        scale=10, scale_units='inches', width=0.01,
                        color='white', edgecolor='black', linewidth=0.5)
        return ax
    
    def plot_action_hist(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(np.array(self.history['r_actor']).T,
                  aspect='auto', cmap='Grays',
                  extent=[0, len(self.history['r_actor'])*self.p.dt, 0, self.p.Nactor],
                  origin='lower')
        t = np.array(self.history['t'])
        ai = np.arctan2(np.array(self.history['a'])[:, 1], np.array(self.history['a'])[:, 0]) / (2 * np.pi) * self.p.Nactor
        ai[ai < 0] += self.p.Nactor
        ax.plot(t+self.p.dt/2, ai+0.5, 'r-', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Actor neuron index')
        ax.set_title('Actor Activity and Action Direction')
        return ax
    
    def plot(self):
        fig, axes = plt.subplots(3,1, figsize=(6, 9))
        self.maze.plot_map(ax=axes[0])
        self.plot_traj(ax=axes[0])
        self.maze.plot_map(ax=axes[1])
        self.plot_valuemap(ax=axes[1])
        self.plot_actormap(ax=axes[1])
        self.plot_action_hist(ax=axes[2])
        plt.tight_layout()
        return fig, axes