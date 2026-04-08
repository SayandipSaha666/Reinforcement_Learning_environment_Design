import gymnasium as gym
import tensorflow as tf
from keras import Model,Input
from keras.layers import Dense
import numpy as np
import cv2

env = gym.make("CartPole-v1", render_mode="rgb_array")

# Q Networks
net_input = Input(shape=(4,)) # state = [cart_position, cart_velocity, pole_angle, pole_velocity]
x = Dense(64,activation='relu')(net_input)
x = Dense(32,activation='relu')(x)
net_output = Dense(2,activation='linear')(x)
q_net = Model(net_input,net_output)

ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
GAMMA = 0.99
NUM_EPISODES = 500

def policy(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(2)
    return tf.argmax(q_net(state)[0]).numpy()

for episodes in range(NUM_EPISODES):
    state = tf.convert_to_tensor([env.reset()[0]])
    done = False
    total_reward = 0
    episode_length = 0
    action = policy(state,EPSILON)

    while not done:
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = tf.convert_to_tensor([next_state])
        next_action = policy(next_state, EPSILON)

        next_q = q_net(next_state)[0]
        target = reward + GAMMA * next_q[next_action]

        if(done):
            target = reward
        
        optimizer = tf.keras.optimizers.Adam(ALPHA)
        with tf.GradientTape() as tape:
            q_values = q_net(state)
            q_val = q_values[0][action]

            loss = (target-q_val)**2

        grads = tape.gradient(loss, q_net.trainable_weights)
        optimizer.apply_gradients(zip(grads, q_net.trainable_weights))

        for j in range(len(grads)):
            q_net.trainable_weights[j].assign_sub(ALPHA * grads[j])
        
        state = next_state
        action = next_action
        total_reward += reward
        episode_length += 1
    print("Episode: ",episodes, "Length: ",episode_length, "Rewards: ", total_reward, "Epsilon: ", EPSILON)
    EPSILON = max(0.01, EPSILON * EPSILON_DECAY)

env.close()
q_net.save('sarsa_q_net.h5')
cv2.destroyAllWindows()