# dqn_mountaincar_gymnasium.py

import random
from collections import deque
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, Input
from keras.models import clone_model
from keras.layers import Dense
from keras.losses import Huber
from keras.optimizers import Adam
import gymnasium as gym

# ========== Environment ==========
env = gym.make("MountainCar-v0")

# ========== Networks ==========
def build_q_network():
    net_input = Input(shape=(2,))
    x = Dense(64, activation="relu")(net_input)
    x = Dense(32, activation="relu")(x)
    output = Dense(env.action_space.n, activation="linear")(x)
    return Model(inputs=net_input, outputs=output)

q_net = build_q_network()
optimizer = Adam(learning_rate=1e-3)
loss_fn = Huber()

# Target network
target_net = clone_model(q_net)
target_net.set_weights(q_net.get_weights())

# ========== Hyperparameters ==========
EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
GAMMA = 0.99
MAX_TRANSITIONS = 100_000
NUM_EPISODES = 200
BATCH_SIZE = 64
TARGET_UPDATE_AFTER = 1000
LEARN_AFTER_STEPS = 4

REPLAY_BUFFER = deque(maxlen=MAX_TRANSITIONS)

# ========== Replay Buffer ==========
def insert_transition(transition):
    REPLAY_BUFFER.append(transition)

def sample_transitions(batch_size=16):
    batch_size = min(batch_size, len(REPLAY_BUFFER))
    sampled = random.sample(REPLAY_BUFFER, batch_size)
    states, actions, rewards, next_states, terminals = zip(*sampled)
    return (
        tf.convert_to_tensor(np.array(states, dtype=np.float32)),
        tf.convert_to_tensor(np.array(actions, dtype=np.int32)),
        tf.convert_to_tensor(np.array(rewards, dtype=np.float32)),
        tf.convert_to_tensor(np.array(next_states, dtype=np.float32)),
        tf.convert_to_tensor(np.array(terminals, dtype=np.bool_)),
    )

# ========== Policy (Epsilon-Greedy) ==========
def policy(state, explore=0.0):
    if random.random() < explore:
        return env.action_space.sample()
    state_tensor = tf.convert_to_tensor(np.expand_dims(state.astype(np.float32), axis=0))
    action = tf.argmax(q_net(state_tensor)[0]).numpy()
    return int(action)

# ========== Utility ==========
def get_q_values(states):
    return tf.reduce_max(q_net(states), axis=1)

# ========== Random States for Q Metric ==========
random_states = []
state, _ = env.reset()
for _ in range(20):
    random_states.append(state)
    action = env.action_space.sample()
    next_state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state
    if done:
        state, _ = env.reset()
random_states = tf.convert_to_tensor(np.array(random_states, dtype=np.float32))

# ========== Training ==========
step_counter = 0
metric = {"episode": [], "length": [], "total_reward": [], "avg_q": [], "exploration": []}

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    total_rewards = 0.0
    episode_length = 0

    while not done:
        action = policy(state, EPSILON)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Modify reward shaping for better learning:
        # Encourage forward progress
        reward += abs(next_state[1]) * 10

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
                batch_nums = tf.range(tf.shape(preds)[0], dtype=tf.int32)
                indices = tf.stack([batch_nums, tf.cast(actions, tf.int32)], axis=1)
                current_values = tf.gather_nd(preds, indices)
                loss = loss_fn(targets, current_values)
            grads = tape.gradient(loss, q_net.trainable_weights)
            optimizer.apply_gradients(zip(grads, q_net.trainable_weights))

        # Update target network
        if step_counter % TARGET_UPDATE_AFTER == 0:
            target_net.set_weights(q_net.get_weights())

    # Logging
    avg_q = float(tf.reduce_mean(get_q_values(random_states)).numpy())
    metric["episode"].append(episode)
    metric["length"].append(episode_length)
    metric["total_reward"].append(total_rewards)
    metric["avg_q"].append(avg_q)
    metric["exploration"].append(EPSILON)

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    if episode % 10 == 0 or episode == NUM_EPISODES - 1:
        pd.DataFrame(metric).to_csv("mountain_metric.csv", index=False)
        print(
            f"Ep {episode:4d} | Len {episode_length:3d} | TotalR {total_rewards:8.2f} "
            f"| AvgQ {avg_q:7.3f} | Eps {EPSILON:5.3f}"
        )

env.close()
q_net.save("dqn_mountaincar_gymnasium.h5")
print("✅ Training complete. Model saved to 'dqn_mountaincar_gymnasium'. Metrics in mountain_metric.csv.")
