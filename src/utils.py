import pandas as pd
from agents import QLearning, EpsilonGreedy
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fourrooms import EnvMode

def train_model(env, lr, gamma, epsilon, nepisodes, nruns, new_goal=None, start_pos=None):
    inital_goal = env.goal
    state, _ = env.reset()
    
    qlearning = QLearning(
        learning_rate= lr,
        gamma = gamma,
        action_size=len(env.action_space),
        state_size=len(env.observation_space)
    )
    policy = EpsilonGreedy(epsilon=epsilon)

    rewards = np.zeros((nepisodes, nruns))
    steps = np.zeros((nepisodes, nruns))
    episodes = np.arange(nepisodes)
    qtables = np.zeros((nruns, len(env.observation_space), qlearning.action_size))
    i=0
    for run in range(nruns):
        env.goal = inital_goal
        qlearning.reset_qtable()
        
        for episode in tqdm(episodes, desc=f"Run {run+1}/{nruns} - Episodes", leave=True):
            if episode == 1000 and new_goal:
                env.goal = new_goal
                print('New goal: ', env.goal)
            state, _ = env.reset()
            if start_pos:
                env.current_cell = start_pos[i]
                i += 1

            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = policy.choose_action(env.action_space, state, qlearning.qtable)

                new_state, reward, terminated, truncated, step = env.step(action)
                done = terminated or truncated

                qlearning.qtable[state, action] = qlearning.update(state, action , reward, new_state)
                total_rewards += reward

                state = new_state

            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = qlearning.qtable
    return qtables, rewards, steps

def postprocess(steps, nepisodes):
    episodes = np.arange(nepisodes)
    st = pd.DataFrame(data={
        "Episodes": episodes,
        "Steps": steps.mean(axis=1),
        "Std": steps.std(axis=1)
    })
    return st

def map_option_to_action(state, action, mode):
    #labels = {0: "UP", 1:"DOWN", 2: "LEFT", 3: "RIGHT", 4:"Hallmark left", 5: "Hallmark right", 6:"Hallmark top", 7:"Hallmark bottom"}

    if mode == EnvMode.OPTIONS:
        action += 4
    if action == 4:
        if state in [25, 88]:
            return 5
        elif state in [51, 62]:
            return 6
        elif is_room1(state) or is_room2(state):
            return 6
        else:
            return 7
    elif action == 5:
        if state in [25, 88]:
            return 4
        elif state in [51, 62]:
            return 7
        elif is_room1(state) or is_room3(state):
            return 4
        else:
            return 5
    return action


def is_room1(field):
    return field < 5 or (field >=10 and field<=14) or (field >=20 and field<=24) or (field >=31 and field<=35)  or (field >=41 and field<=45)

def is_room2(field):
    return (field >= 5 and field <=9) or (field >=15 and field<=19) or (field >=26 and field<=30) or (field >=36 and field<=40) or (field >=46 and field<=50) or (field >=52 and field<=56)

def is_room3(field):
    return (field >= 57 and field <=61) or (field >=63 and field<=67) or (field >=73 and field<=77) or (field >=83 and field<=87) or (field >=94 and field<=98)


def qtable_visualisation(env, qtable, map_size=13):

    qtable_max = qtable.max(axis=1).flatten()#.reshape(map_size, map_size)
    qtable_val_max = env.occupancy.copy()
    qtable_val_max[qtable_val_max == 0] = 2
    qtable_val_max[qtable_val_max == 1] = np.finfo(float).eps
    qtable_val_max = qtable_val_max.flatten().astype(float)
    qtable_best_action = np.argmax(qtable, axis=1).flatten()#.reshape(map_size, map_size)
    for i in range(len(qtable_best_action)):
        qtable_best_action[i] = map_option_to_action(i, qtable_best_action[i], env.mode)

    directions = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "L", 5: "R", 6: "T", 7:"B"}
    qtable_directions = np.empty(map_size*map_size, dtype=str)

    eps = np.finfo(float).eps
    i = 0
    for idx, val in enumerate(qtable_val_max):
        if val == 2:
            qtable_val_max[idx] = qtable_max[i]
            if qtable_max[i] > eps:
                qtable_directions[idx] = directions[qtable_best_action[i]]
            i += 1
    qtable_val_max = qtable_val_max.reshape((map_size, map_size))
    qtable_directions = qtable_directions.reshape((map_size, map_size))
    return qtable_val_max, qtable_directions

def plot_q_values_map(qtable, env, map_size=13):
    qtable_val_max, qtable_directions = qtable_visualisation(env, qtable, map_size)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    ax[0].imshow(env.render(), cmap="Blues")
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor ="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"}
    ).set(title="Learned Q-values \nArrows respresent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    plt.show()
