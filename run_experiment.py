import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from Morris_spike import ACMaze


pf = "res"
os.makedirs(pf, exist_ok=True)

results = pd.DataFrame(columns=["session", "trial", "reached", "reached_time"])


starts = [
    (0.0, +7.5),
    (0.0, -7.5),
    (-7.5, 0.0),
    (+7.5, 0.0),
]

Nsessions = 20; Ntrials = 100
for session in range(Nsessions):

    agent = ACMaze(seed=33+session)

    for trial_i in range(Ntrials):
        start = starts[agent.rng.integers(0, len(starts))]

        print(f"Sesssion {session+1}/{Nsessions}; Trial {trial_i+1}/{Ntrials}: starting at {start}")

        out = agent.run_trial(start)

        fig, axes = agent.plot()
        plt.savefig(f'{pf}/session_{session+1}_trial_{trial_i+1:03d}.png', dpi=200)
        plt.close()

        results.loc[len(results)] = {
            "session": session,
            "trial": trial_i,
            "reached": out["reached"],
            "reached_time": out["reached_time"] if out["reached"] else agent.p.Ttimeout,
        }
        
    results.to_csv(f'{pf}/results.csv', index=False)
    
    fig_tg, ax_tg = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=results, x='trial', y='reached_time', ax=ax_tg)
    ax_tg.set_xlabel('Trial')
    ax_tg.set_ylabel('Time to Goal or Timeout (s)')
    ax_tg.set_title(f'Time to Goal over Trials (sessions 1-{session+1})')
    ax_tg.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{pf}/time_to_goal_over_trials.png', dpi=200)
    plt.close(fig_tg)

print()
print("=" * 60)
print("Test Complete!")
print("=" * 60)

