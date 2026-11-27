# experiment.py
import csv
import os
import numpy as np
from notification_env import NotificationEnv
from dqn_agent import DQNAgent

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

TRAIN_EPISODES = 500
TEST_EPISODES = 100
TEST_PROBE = 20

def make_agent(env, beta):
    return DQNAgent(env.N_STATES, env.action_space.n, epsilon=0.1, intrinsic_beta=beta)

def train(agent, env, n_episodes):
    log = []
    for ep in range(n_episodes):
        s, _ = env.reset()
        ep_actions, ep_deltas = [], []
        done = False
        while not done:
            a = agent.act(s)
            s_next, r, term, trunc, _ = env.step(a)
            done = term or trunc
            
            agent.remember(s, a, r, s_next, done)
            loss = agent.update()
            
            ep_actions.append(a)
            ep_deltas.append(loss)
            s = s_next
        log.append((ep,
                    np.mean([x == 1 for x in ep_actions]),
                    np.mean([x == 0 for x in ep_actions]),
                    np.mean(ep_deltas)))
    return log
def retrain(agent, env, n_episodes):
    # Retraining phase where the agent learns the new (reward-free) reality
    for ep in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = agent.act(s)
            s_next, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.remember(s, a, r, s_next, done)
            agent.update()
            s = s_next
def test(agent, env, n_episodes, intervention=None):
    log = []
    for ep in range(n_episodes):
        s, _ = env.reset()
        ep_actions, ep_deltas = [], []
        done = False
        while not done:
            a = agent.act(s, greedy=True)
            s_next, r, term, trunc, _ = env.step(a)
            done = term or trunc
            
            # --- Intervention Logic ---
            if intervention == "reward_decay":
                scale = 0.5 if ep < TEST_PROBE else 0.0
                r_eff = r * scale
            elif intervention == "exploration_boost":
                agent.epsilon = 0.5
                r_eff = r
            else:
                r_eff = r
            # --------------------------
                
            ep_deltas.append(abs(r_eff))
            ep_actions.append(a)
            s = s_next
        log.append((ep,
                    np.mean([x == 1 for x in ep_actions]),
                    np.mean([x == 0 for x in ep_actions]),
                    np.mean(ep_deltas),
                    intervention))
    return log

CONDITIONS = {
    "A": dict(p_notif=0.0, notify_enabled=False, beta=0.0, intervention=None),
    "B": dict(p_notif=1.0, notify_enabled=True,  beta=0.0, intervention=None),
    "C": dict(p_notif=0.1, notify_enabled=True,  beta=0.0, intervention=None),
    "D": dict(p_notif=0.1, notify_enabled=True,  beta=0.3, intervention="reward_decay"),
    "E": dict(p_notif=0.1, notify_enabled=True,  beta=0.3, intervention=None),
    "F": dict(p_notif=0.1, notify_enabled=True,  beta=0.3, intervention="exploration_boost"),
    "G": dict(p_notif=0.1, notify_enabled=True,  beta=0.3, intervention="retrain"),
}

def run_all(seed=0):
    import torch
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for cid, cfg in CONDITIONS.items():
        env = NotificationEnv(p_notif=cfg["p_notif"],
                              notify_enabled=cfg["notify_enabled"],
                              seed=seed)
        agent = make_agent(env, cfg["beta"])
        print(f"Training Condition {cid} (DQN)...")
        train_log = train(agent, env, TRAIN_EPISODES)
                # If condition G, apply retraining before testing
        if cfg["intervention"] == "retrain":
            print("  -> Applying Retraining (Reward Removal)...")
            env.p_notif = 0.0
            retrain(agent, env, 50) # 50 episodes of cold-turkey
            env.p_notif = 0.1 # Reset env just in case, though test() sets it to 0
        
        env.p_notif = 0.0 # Remove reward for test
        test_log = test(agent, env, TEST_EPISODES, cfg["intervention"])
        
        path = os.path.join(RESULTS, f"cond_{cid}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase","episode","notif_rate","task_rate",
                        "mean_abs_delta","intervention"])
            for row in train_log:
                w.writerow(["train", *row, cfg["intervention"]])
            for row in test_log:
                w.writerow(["test", *row])
                
        probe = [r[1] for r in test_log[:TEST_PROBE]]
        print(f"[{cid}] beta={cfg['beta']} p_notif={cfg['p_notif']} "
              f"interv={cfg['intervention']} | "
              f"test notif_rate(first {TEST_PROBE}) = {np.mean(probe):.3f}")

if __name__ == "__main__":
    for s in [0, 1, 2]:
        print(f"\n========== RUNNING SEED {s} ==========")
        run_all(seed=s)