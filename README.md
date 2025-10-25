# Reinforcement Learning

## Code for Part 5

In this i learned Deep Q-Learning and retrained our CartPole problem with this. I also trained an agent on the Mountain Car Problem.

##### Training Results


<img src="https://user-images.githubusercontent.com/53657825/178180071-bb173c3a-d510-4af9-8b22-78de2b1ff7d2.gif" width="300" height="200">  <img src="https://user-images.githubusercontent.com/53657825/178180140-2c86cdc4-4153-4d89-9891-8f0af3460955.gif" width="300" height="200">

# 🧠 Deep Q-Network (DQN)

## 📌 Overview
**Deep Q-Network (DQN)** is a reinforcement learning algorithm that combines **Q-learning** with **deep neural networks** to solve environments with large or continuous state spaces.  
Instead of using a traditional Q-table to store state-action values, DQN uses a neural network to approximate the **Q-function**:
\[
Q(s, a; \theta) \approx Q^*(s, a)
\]
This allows the agent to learn optimal actions directly from high-dimensional inputs such as environment states, pixels, or sensor data.

---

## ⚙️ Core Concepts

### 🧩 Q-Learning Recap
Q-learning aims to learn the optimal action-value function:
\[
Q^*(s, a) = r + \gamma \max_{a'} Q^*(s', a')
\]
The goal is to maximize the expected cumulative reward.

### 🤖 Neural Network as Function Approximator
In DQN, a **neural network** replaces the Q-table and outputs the Q-values for all possible actions given a state.  
This helps generalize across unseen states and reduces the need for explicit storage of Q-values.

---

## 🚀 DQN Architecture Components

### 1. **Replay Buffer**
A replay buffer (experience replay) stores past transitions `(state, action, reward, next_state, done)`.  
During training, random mini-batches are sampled from it to:
- Break correlations between consecutive samples
- Improve learning stability

### 2. **Target Network**
A secondary network (`target_net`) used to compute target Q-values.  
It’s updated periodically with weights from the main Q-network (`q_net`) to prevent oscillations and improve convergence.

### 3. **Epsilon-Greedy Policy**
Used to balance **exploration** and **exploitation**:
- With probability ε → take a **random** action
- With probability (1−ε) → take the **best** action according to current Q-values
- ε decays over time to reduce exploration as learning progresses

### 4. **Loss Function**
DQN minimizes the error between predicted and target Q-values using the **Huber Loss**:
\[
L = (Q_{\text{target}} - Q(s, a))^2
\]
where:
\[
Q_{\text{target}} = r + \gamma \max_{a'} Q_{\text{target}}(s', a')
\]

---

## 🧠 Algorithm Steps
1. Initialize replay buffer and Q-network with random weights.  
2. For each episode:
   - Get the current state `s`.
   - Choose an action `a` using the epsilon-greedy policy.
   - Perform the action, observe `r`, `s'`, and `done`.
   - Store `(s, a, r, s', done)` in replay buffer.
   - Sample random batches and compute target Q-values using target network.
   - Train the Q-network using gradient descent.
   - Periodically update the target network weights.
3. Repeat until the agent achieves optimal performance.

---

## 🧮 Mathematical Summary
**Bellman Equation:**
\[
Q(s, a) = r + \gamma \max_{a'} Q(s', a')
\]

**Network Training Objective:**
\[
L(\theta) = \mathbb{E}\big[(Q_{\text{target}} - Q(s, a; \theta))^2\big]
\]

---

## 🎯 Applications
- **Atari & Classic Control Games** (CartPole, MountainCar, Breakout)
- **Autonomous Driving**
- **Robotics and Manipulation**
- **Stock Market & Trading Agents**
- **Smart Grid / Energy Optimization**

---

## ✅ Advantages
- Handles **large and continuous state spaces**
- Learns **directly from raw sensory input (e.g., pixels)**
- Uses **experience replay** for stable learning
- Incorporates **target networks** for smoother convergence

---

## ⚠️ Limitations
- Struggles with **continuous action spaces** (addressed by DDPG, PPO, etc.)
- **Sensitive to hyperparameters** (learning rate, batch size, update frequency)
- Can **diverge** without proper tuning and normalization

---

## 🧰 Example Environments
- **CartPole-v1**  
- **MountainCar-v0**  
- **LunarLander-v2**

---

## 📈 Typical Results
With sufficient training:
- CartPole achieves balance for >500 steps.
- MountainCar learns to reach the goal efficiently.
- Q-values stabilize, and total episode reward increases steadily.

---

## 🧾 References
- Mnih et al., *“Playing Atari with Deep Reinforcement Learning”*, DeepMind (2013)
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd Edition)
- OpenAI Gymnasium Environments Documentation

---

