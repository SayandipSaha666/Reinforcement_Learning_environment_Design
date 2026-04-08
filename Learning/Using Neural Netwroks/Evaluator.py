import gymnasium as gym
import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np

# Create env with render mode
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Load trained model
q_net = load_model("sarsa_q_net.h5")

def policy(state, explore=0.0):
    if np.random.random() < explore:
        return np.random.randint(2)
    
    q_values = q_net(state)
    return tf.argmax(q_values[0]).numpy()


for episode in range(500):
    done = False

    # Reset environment
    state, _ = env.reset()
    state = tf.convert_to_tensor([state], dtype=tf.float32)

    while not done:
        # Get frame (no mode argument here)
        frame = env.render()
        cv2.imshow("CartPole", frame)
        cv2.waitKey(50)

        action = policy(state)

        # Step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = tf.convert_to_tensor([next_state], dtype=tf.float32)

env.close()
cv2.destroyAllWindows()