import gym
import random
from IPython.display import clear_output
import numpy as np
import os
from time import sleep

os.environ["SDL_VIDEODRIVER"] = "dummy"
env = gym.make("Taxi-v3")
env.reset()
print(env.render())
print(env.action_space)
print(env.observation_space)
alpha = 0.1
gamma = 0.6
epsilon = 0.1
q_table = np.zeros([env.observation_space.n, env.action_space.n])
print(q_table.shape)
print(q_table)
for i in range(100000):  # episodes / game
    state = env.reset()

    penalties, reward = 0, 0
    done = False
    while not done:
        # Exploration
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        # Exploitation
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        q_old = q_table[state, action]
        next_max = np.max(q_table[next_state])

        q_new = (1 - alpha) * q_old + alpha * (reward + gamma * next_max)
        q_table[state, action] = q_new

        if reward == -10:
            penalties += 1

        state = next_state

    if i % 100 == 0:
        clear_output(wait=True)
        print("Episode: ", i)
print("The training has finished!")
total_penalidade = 0
episodios = 50
frames = []

for _ in range(episodios):
    estado = env.reset()
    penalties, recompensa = 0, 0
    done = False
    while not done:
        acao = np.argmax(q_table[state])
        state, recompensa, done, info = env.step(action=acao)
        if recompensa == -10:
            penalties += 1
        frames.append(
            {
                "frame": env.render(mode="ansi"),
                "state": estado,
                "action": acao,
                "reward": recompensa,
            }
        )
    total_penalidade += penalties
print("episodios", episodios, "penalidades", total_penalidade)
for frame in frames:
    clear_output(wait=True)
    print(frame["frame"])
    print("Estado", frame["state"])
    print("Acao", frame["action"])
    print("Recompensa", frame["reward"])
    sleep(0.1)
