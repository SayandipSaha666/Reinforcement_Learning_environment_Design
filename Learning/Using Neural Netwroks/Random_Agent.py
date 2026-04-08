import cv2
import gymnasium as gym
import tensorflow as tf

env = gym.make("CartPole-v1", render_mode="rgb_array")

for episode in range(5):
    state, info = env.reset()
    done = False

    while not done:
        frame = env.render()

        cv2.imshow("CartPole", frame)
        cv2.waitKey(100)

        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32).numpy()
        state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

env.close()
cv2.destroyAllWindows()