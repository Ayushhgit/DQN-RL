# dqn_cartpole_gymnasium.py

import random
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, Input
from keras.models import clone_model
from keras.layers import Dense
from keras.losses import Huber
from keras.optimizers import Adam

# ---- Environment ----
env = gym.make("CartPole-v1")

# ---- Hyperparameters ----
EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995          # multiplicative decay
GAMMA = 0.99
MAX_TRANSITIONS = 100_000
NUM_EPISODES = 200
BATCH_SIZE = 64
TARGET_UPDATE_AFTER = 1000     # steps between target updates
LEARN_AFTER_STEPS = 1          # how often to learn
LR = 1e-3

# ---- Replay Buffer ----
REPLAY_BUFFER = deque(maxlen=MAX_TRANSITIONS)

# ---- Build Q-Network ----
def build_q_network():
    net_input = Input(shape=(4,))
    x = Dense(32, activation="relu")(net_input)
    x = Dense(16, activation="relu")(x)
    output = Dense(env.action_space.n, activation="linear")(x)
    return Model(inputs=net_input, outputs=output)

q_net = build_q_network()
optimizer = Adam(learning_rate=LR)
loss_fn = Huber()

# Target network
target_net = clone_model(q_net)
target_net.set_weights(q_net.get_weights())

# ---- Utility Functions ----
def insert_transition(transition):
    REPLAY_BUFFER.append(transition)

def sample_transitions(batch_size=BATCH_SIZE):
    batch_size = min(batch_size, len(REPLAY_BUFFER))
    sampled = random.sample(REPLAY_BUFFER, batch_size)
    states, actions, rewards, next_states, terminals = zip(*sampled)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    terminals = np.array(terminals, dtype=np.bool_)
    return (
        tf.convert_to_tensor(states),
        tf.convert_to_tensor(actions),
        tf.convert_to_tensor(rewards),
        tf.convert_to_tensor(next_states),
        tf.convert_to_tensor(terminals),
    )

def policy(state, explore=0.0):
    """Epsilon-greedy policy"""
    if random.random() < explore:
        return env.action_space.sample()
    state_tensor = tf.convert_to_tensor(np.expand_dims(state.astype(np.float32), axis=0))
    action = tf.argmax(q_net(state_tensor)[0]).numpy()
    return int(action)

def calculate_reward(state):
    """Custom shaped reward (optional)"""
    reward = -1.0
    if -0.5 <= state[0] <= 0.5 and -1 <= state[1] <= 1 and -0.07 <= state[2] <= 0.07 and -0.525 <= state[3] <= 0.525:
        reward = 1.0
    return reward

def get_q_values(states):
    return tf.reduce_max(q_net(states), axis=1)

# ---- Collect random states for avg Q metric ----
random_states = []
state, _ = env.reset()
for _ in range(20):
    random_states.append(state)
    action = policy(state, explore=1.0)
    next_state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state
    if done:
        state, _ = env.reset()
random_states = tf.convert_to_tensor(np.array(random_states, dtype=np.float32))

# ---- Training Loop ----
step_counter = 0
metric = {"episode": [], "length": [], "total_reward": [], "avg_q": [], "exploration": []}

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    total_rewards = 0.0
    episode_length = 0

    while not done:
        action = policy(state, explore=EPSILON)
        next_state, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # You can switch between custom and env reward
        reward = calculate_reward(next_state)
        # reward = env_reward  # uncomment for default environment reward

        insert_transition((state.copy(), action, reward, next_state.copy(), done))
        state = next_state
        total_rewards += reward
        episode_length += 1
        step_counter += 1

        # Learn
        if len(REPLAY_BUFFER) >= BATCH_SIZE and step_counter % LEARN_AFTER_STEPS == 0:
            current_states, actions, rewards, next_states, terminals = sample_transitions(BATCH_SIZE)
            next_action_values = tf.reduce_max(target_net(next_states), axis=1)
            terminals = tf.cast(terminals, tf.bool)
            targets = tf.where(terminals, rewards, rewards + GAMMA * next_action_values)

            with tf.GradientTape() as tape:
                preds = q_net(current_states)
                batch_nums = tf.range(0, tf.shape(preds)[0], dtype=tf.int32)
                indices = tf.stack([batch_nums, tf.cast(actions, tf.int32)], axis=1)
                current_values = tf.gather_nd(preds, indices)
                loss = loss_fn(targets, current_values)

            grads = tape.gradient(loss, q_net.trainable_weights)
            optimizer.apply_gradients(zip(grads, q_net.trainable_weights))

        # Target update
        if step_counter % TARGET_UPDATE_AFTER == 0:
            target_net.set_weights(q_net.get_weights())

    # ---- End of Episode ----
    avg_q = float(tf.reduce_mean(get_q_values(random_states)).numpy())
    metric["episode"].append(episode)
    metric["length"].append(episode_length)
    metric["total_reward"].append(total_rewards)
    metric["avg_q"].append(avg_q)
    metric["exploration"].append(EPSILON)

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    # Save metrics every 10 episodes
    if episode % 10 == 0 or episode == NUM_EPISODES - 1:
        pd.DataFrame(metric).to_csv("metric.csv", index=False)
        print(
            f"Episode {episode:4d} | len {episode_length:3d} | total_reward {total_rewards:.2f} | "
            f"avg_q {avg_q:.3f} | eps {EPSILON:.3f}"
        )

# ---- Save and Cleanup ----
env.close()
q_net.save("dqn_q_net_gymnasium.h5")
print("✅ Training finished. Model saved as 'dqn_q_net_gymnasium'. Metrics saved to metric.csv.")
