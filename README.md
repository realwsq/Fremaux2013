# Continuous-Time Actor-Critic with Spiking Neurons — Morris Water Maze

Implementation of the reinforcement learning model from:

> Frémaux N., Sprekeler H., Gerstner W. (2013).
> **Reinforcement Learning Using a Continuous Time Actor-Critic Framework with Spiking Neurons.**
> *PLOS Computational Biology*, 9(4): e1003024.
> https://doi.org/10.1371/journal.pcbi.1003024

This project implements a biologically inspired continuous-time actor-critic model using spiking neurons. The agent learns to navigate a 20 × 20 square maze containing a U-shaped obstacle to reach a hidden goal located at the origin.
Place cells encode position as Poisson spikes; 
a critic network (100 SRM0 neurons) learns the value function V(x); 
an actor network (180 SRM0 neurons with a ring attractor) outputs a velocity direction.  
Weights are updated online by the TD-LTP rule (Eq. 17) using a continuous-time TD error.

This implementation qualitatively reproduces the main results reported in the original publication. Further performance improvements could be achieved through systematic hyperparameter optimization.

---

## Dependencies

Python ≥ 3.9 and the following packages:

```bash
pip install numpy matplotlib seaborn pandas tqdm
```

Or with conda:

```bash
conda install numpy matplotlib seaborn pandas tqdm
```

Tested with Python 3.10/3.11, numpy 1.24+, matplotlib 3.7+.

---

## Directory structure

```
watermaze_spike/
├── run_experiment.py                               # Entry point: runs 20 sessions (i.e., seeds) × 100 trials, saves results to res/
├── Morris_spike.py                                 # Core model: Params, ACMaze (place cells, SRM0, actor, critic, TD-LTP)
├── maze.py                                         # Maze geometry (Maze class): U-obstacle, goal, wall bounce
├── utils.py                                        # Double-exponential filter utilities 
└── res0/                                           # Reference results (20 sessions (i.e., seeds) × 100 trials)
    ├── time_to_goal_over_trials.png                # Learning curve averaged all sessions
    ├── session_1_trial_001.png … trial_010.png     # How agent behaves before learning (example session 1)
    └── session_1_trial_091.png … trial_100.png     # How agent behaves after learning (example session 1)
```

---

## Reproduce results

```bash
python run_experiment.py
```

Output is written to `res/`.  A full run (20 sessions × 100 trials) takes several hours on a single CPU core.  To reproduce only session 0 for a quick check, set `Nsessions = 1` in `run_experiment.py`.

Expected learning curve: time-to-goal drops from ~50 s (timeout) to steadily reaching the goal within roughly 20-40 trials, consistent with `res0/time_to_goal_over_trials.png`.

---

## Output files

| File | Description |
|------|-------------|
| `res/session_X_trial_NNN.png` | 3-panel diagnostic for trial NNN: trajectory, value map + actor arrows, actor ring activity |
| `res/time_to_goal_over_trials.png` | Learning curve: mean time-to-goal (or 50 s timeout) vs. trial number, aggregated over all sessions |
