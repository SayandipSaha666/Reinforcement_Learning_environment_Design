import gymnasium as gym
import numpy as np
import pickle as pkl

cliffEnv = gym.make("CliffWalking-v1")

q_table = np.zeros(shape=(48,4))

# Epsilon greedy Policy
def policy(state, explore=0.0):
    if np.random.random() < explore:
        return np.random.randint(0, 4)
    return int(np.argmax(q_table[state]))

GAMMA=0.9
ALPHA=0.1
NUM_EPISODES = 500

for episode in range(NUM_EPISODES):
    
    state, info = cliffEnv.reset()
    done = False
    total_reward = 0
    episode_length = 0

    action = policy(state)

    while not done:
        next_state, reward, terminated, truncated, info = cliffEnv.step(action)
        done = terminated or truncated

        next_action = policy(next_state)

        if done:
            target = reward
        else:
            target = reward + GAMMA * q_table[next_state][next_action]

        q_table[state][action] += ALPHA * (target - q_table[state][action])

        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1

    print(f"Episode {episode}, Length {episode_length}, Reward {total_reward}")

cliffEnv.close()

pkl.dump(q_table,open("q_learning_q_table.pkl","wb"))
print("Training Complete. Q Table Saved")