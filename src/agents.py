import numpy as np
from tqdm import tqdm

class QLearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.qtable = np.zeros((self.state_size, self.action_size))


    def update(self, state, action, reward, new_state):
        """Update Q(s, a) = Q(s, a) + lr [R(s,a) +  gamma * max Q(s', a') - Q(s, a)]"""
        delta = (
            reward 
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
            )
        return self.qtable[state, action] + self.learning_rate * delta
    
    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))
    
class EpsilonGreedy:
    def __init__(self, epsilon) -> None:
        self.epsilon = epsilon

    def choose_action(self, action_space, state, qtable, is_training=True):
        if is_training:
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(action_space,1)[0]
                return action

        best_actions = np.argwhere(qtable[state, :] == np.amax(qtable[state, :])).flatten()
        if len(best_actions) > 1:
            action = np.random.choice(best_actions, 1)[0]
        else:
            action = best_actions[0]

        return action

class EpsGreedyPolicy():

	def __init__(self, rng, nstates, noptions, epsilon):
		self.rng = rng
		self.nstates = nstates
		self.noptions = noptions
		self.epsilon = epsilon
		self.Q_Omega_table = np.zeros((nstates, noptions))

	def Q_Omega(self, state, option=None):
		if option is None:
			return self.Q_Omega_table[state,:]
		else:
			return self.Q_Omega_table[state, option]

	def predict(self, state):
		if self.rng.uniform() < self.epsilon:
			return int(self.rng.randint(self.noptions))
		else:
			return int(np.argmax(self.Q_Omega(state)))

class BoltzmannPolicy():

	def __init__(self, rng, lr, nstates, nactions, temperature=1.0):
		self.rng = rng
		self.lr = lr
		self.nstates = nstates
		self.nactions = nactions
		self.temperature = temperature
		self.weights = np.zeros((nstates, nactions))

	def Q_U(self, state, action=None):
		if action is None:
			return self.weights[state,:]
		else:
			return self.weights[state, action]

	def boltzmann(self, state):
		exp_probabilities = np.exp(self.Q_U(state) / self.temperature)
		return exp_probabilities / np.sum(exp_probabilities)

	def predict(self, state):
		return self.rng.choice(self.nactions, p=self.boltzmann(state))

	def gradient_step(self, state, action, Q_U):
		probabilities = self.boltzmann(state)
		self.weights[state, :] -= self.lr * probabilities * Q_U
		self.weights[state, action] += self.lr * Q_U


class SigmoidFunction():

	def __init__(self, lr, nstates):
		self.lr = lr
		self.nstates = nstates
		self.weights = np.zeros((nstates,))

	def sigmoid(self,state):
		return 1/ (1+np.exp(-self.weights[state]))
	
	def predict(self, state):
		prob_termination = self.sigmoid(state)
		return np.random.choice([0, 1], 1, p=[1-prob_termination, prob_termination])

	def derivative(self, state):
		return self.sigmoid(state) * (1.0 - self.sigmoid(state))

	def gradient_step(self, state, A):
		derivative = self.derivative(state)
		self.weights[state] -= self.lr * derivative * A

class OptionCritic:
	def __init__(self, rng, temperature, epsilon, lr_intra, lr_term, env, nsteps,lr_critic, discount, noptions):
		self.env = env
		self.nstates = env.observation_space.shape[0]
		self.nactions = env.action_space.shape[0]
		self.nsteps = nsteps
		self.lr_critic = lr_critic
		self.lr_intra = lr_intra
		self.lr_term = lr_term
		self.rng = rng
		self.lr_critic = lr_critic
		self.discount = discount
		self.noptions = noptions
		self.epsilon = epsilon
		self.temperature = temperature
    
	def train(self, nruns, nepisodes, possible_next_goals):
		self.goals = []
		self.start_pos = []
		self.steps = np.zeros((nepisodes, nruns))
		self.option_terminations_list = []
		for run in range(nruns):
			self.actor = Actor(self.rng, self.lr_intra, self.lr_term, self.nstates, self.nactions, self.noptions, self.temperature, self.epsilon)
			self.critic = Critic(self.lr_critic, self.discount, self.actor.policy_over_options.Q_Omega_table, self.nstates, self.noptions, self.nactions)			
			self._execute_run(run, self.rng, nepisodes, possible_next_goals)
		return self.goals, self.start_pos, self.steps

	def _execute_run(self, run, rng, nepisodes, possible_next_goals):
		episodes = np.arange(nepisodes)
		for episode in tqdm(episodes, desc=f"Run {run+1} - Episodes", leave=True):
			if episode != 0 and episode % 1000 == 0 and possible_next_goals:
				self.env.goal = rng.choice(possible_next_goals)
				self.goals.append(self.env.goal)
				print('New goal:', self.env.goal)
				
			state, _ = self.env.reset()
			self.start_pos.append(self.env.current_cell)

			option = self.actor.select_option(state)
			action = self.actor.select_action(option, state)
			self.critic.cache(state, option, action)

			duration = 1
			option_switches = 0
			avg_duration = 0.0

			for step in range(self.nsteps):
				
				state, reward, done, truncated, _ = self.env.step(action)

				if self.actor.predict_termination(option, state):
					option = self.actor.select_option(state)
					option_switches += 1
					avg_duration += (1.0/option_switches)*(duration - avg_duration)
					duration = 1
					
				action = self.actor.select_action(option, state)
              
				
				self.critic.update_Qs(state, option, action, reward, done, self.actor.option_terminations)
				A_Omega = self.critic.A_Omega(state, option)
				Q_U = self.critic.Q_U(state, option, action) - self.critic.Q_Omega(state, option)
				
				self.actor.update_policies(state, option, action, Q_U, A_Omega)
				
				duration += 1

				if done or truncated:
					break

			self.steps[episode, run] = step
		self.option_terminations_list.append(self.actor.option_terminations)


class Actor():
    def __init__(self, rng, lr_intra, lr_term, nstates, nactions, noptions, temperature, epsilon):
        self.option_policies = [BoltzmannPolicy(rng, lr_intra, nstates, nactions, temperature) for _ in range(noptions)]
        self.option_terminations = [SigmoidFunction(lr_term, nstates) for _ in range(noptions)]
        self.policy_over_options = EpsGreedyPolicy(rng, nstates, noptions, epsilon)

    def select_option(self, state):
        return self.policy_over_options.predict(state)

    def select_action(self, option, state):
        return self.option_policies[option].predict(state)
	
    def predict_termination(self, option, state):
        return self.option_terminations[option].predict(state)

    def update_policies(self, state, option, action, Q_U, A_Omega):
        self.option_policies[option].gradient_step(state, action, Q_U)
        self.option_terminations[option].gradient_step(state, A_Omega)

class Critic():

	def __init__(self, lr, discount, Q_Omega_table, nstates, noptions, nactions):
		self.lr = lr
		self.discount = discount
		self.Q_Omega_table = Q_Omega_table
		self.Q_U_table = np.zeros((nstates, noptions, nactions))

	def cache(self, state, option, action):
		self.last_state = state
		self.last_option = option
		self.last_action = action
		self.last_Q_Omega = self.Q_Omega(state, option)

	def Q_Omega(self, state, option=None):
		if option is None:
			return self.Q_Omega_table[state, :]
		else:
			return self.Q_Omega_table[state, option]

	def Q_U(self, state, option, action):
		return self.Q_U_table[state, option, action]

	def A_Omega(self, state, option=None):
		advantage = self.Q_Omega(state) - np.max(self.Q_Omega(state))

		if option is None:
			return advantage
		else:
			return advantage[option]

	def update_Qs(self, state, option, action, reward, done, terminations):
		target = reward
		if not done:
			beta_omega = terminations[self.last_option].sigmoid(state)
			target += self.discount * ((1.0 - beta_omega)*self.Q_Omega(state, self.last_option) + \
						beta_omega*np.max(self.Q_Omega(state)))

		tderror_Q_Omega = target - self.last_Q_Omega
		self.Q_Omega_table[self.last_state, self.last_option] += self.lr * tderror_Q_Omega

		tderror_Q_U = target - self.Q_U(self.last_state, self.last_option, self.last_action)
		self.Q_U_table[self.last_state, self.last_option, self.last_action] += self.lr * tderror_Q_U

		self.last_state = state
		self.last_option = option
		self.last_action = action
		if not done:
			self.last_Q_Omega = self.Q_Omega(state, option)