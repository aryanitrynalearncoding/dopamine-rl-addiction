# Dopamine-Inspired Reward Prediction Error in Deep RL Agents

## 📌 Project Overview
This project investigates whether an artificial agent with a dopamine-inspired Reward Prediction Error (RPE) mechanism can develop persistent, compulsive reward-seeking behavior (analogous to smartphone notification checking).

Biological dopamine is not merely a "pleasure" chemical; it encodes Reward Prediction Error (RPE)—the difference between expected and received rewards. Standard Reinforcement Learning (RL) agents use RPE as a learning signal but maximize expected value rationally. This project explores what happens when we add an *intrinsic* bonus for high RPE (simulating the "thrill of the unknown"), and whether that behavior can be extinguished.

## 🧠 Environment Design
We use a custom Gymnasium environment (`NotificationEnv`). The agent sits at a desk with three possible actions:
1. **Do Task (Action 0):** Guaranteed, predictable reward of `+5`.
2. **Check Notification (Action 1):** Intermittent, unpredictable reward. 10% chance of `+20`, 90% chance of `0`.
3. **Idle (Action 2):** Reward of `0`.

Mathematically, the expected value of checking the notification is `0.1 * 20 = 2`, which is worse than doing the task (`5`). Therefore, a rational agent should ignore the notification.

## 🧪 Experiment Conditions (A–G)
We train Deep Q-Networks (DQN) under seven different conditions to isolate the emergence, resistance, and cure for reward-driven persistence:

*   **A (Baseline):** No notification action available.
*   **B (Constant Reward):** Notification always gives `+20`.
*   **C (Rational DQN):** Intermittent reward, standard Q-learning. (Agent correctly learns to ignore the notification).
*   **E (Dopamine DQN):** Intermittent reward with an intrinsic RPE bonus (\(R_{learn} = R_{ext} + \beta|\delta|\)). Agent becomes severely addicted.
*   **D (Intervention - Reward Decay):** Applied to the addicted agent. The notification reward is devalued during testing.
*   **F (Intervention - Exploration Boost):** Applied to the addicted agent. The agent's exploration rate (\(\epsilon\)) is forced to 0.5 during testing.
*   **G (Intervention - Retraining):** The addicted agent is forced to undergo 50 episodes of "cold turkey" retraining in a reward-free environment before testing.

## 📊 Key Findings
Tested across 3 independent random seeds, the results demonstrate:
1.  **Rationality (C):** Standard RL agents mathematically ignore unpredictable, low-expectation rewards (`0.000` notification rate).
2.  **Emergence (E):** Adding an intrinsic RPE bonus causes immediate, severe addiction (`~0.970` notification rate), overriding rational expected-value maximization.
3.  **Resistance (D & F):** Superficial test-time interventions (reward decay or forced exploration) are highly unreliable. The neural network's weights are locked into the compulsive behavior.
4.  **The Cure (G):** Only direct retraining (unlearning through new experience) consistently breaks the addiction across all seeds (`0.000` notification rate). 

*Conclusion:* Dopamine-inspired RPE mechanisms are sufficient to generate compulsive reward-seeking in Deep RL. These behaviors are robust to standard behavioral manipulations and require fundamental policy unlearning to resolve.

## 📂 Repository Structure
*   `notification_env.py`: The custom Gymnasium environment.
*   `dqn_agent.py`: PyTorch implementation of the Deep Q-Network with the intrinsic RPE bonus.
*   `experiment.py`: Script to run all 7 experimental conditions across multiple seeds.
*   `analyze.py`: Script to generate the training/extinction graphs (`results/summary.png`).
*   `results/`: Contains the generated CSVs and summary graphs.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/aryanitrynalearncoding/dopamine-rl-addiction.git
   cd dopamine-rl-addiction
