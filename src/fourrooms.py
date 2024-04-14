import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from enum import Enum

class EnvMode(Enum):
	TRIVIAL = 1
	OPTIONS = 2
	ALL = 3

class FourRooms:

	def __init__(self, max_steps=1000, mode=EnvMode.TRIVIAL, start_pos=None, goal_pos=None, rendering=False):
		layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
		self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
		self.steps = 0
		self.max_steps = max_steps
		self.rendering = rendering
		# Doorway Left (6, 2)
		# Doorway Down (10, 6)
        # Doorway Right (7, 9)
        # Doorway Up (3, 6)
		
		# Four possible actions
		# 0: UP
		# 1: DOWN
		# 2: LEFT
		# 3: RIGHT
		if mode == EnvMode.TRIVIAL:
			self.action_space = np.array([0, 1, 2, 3])
		elif mode == EnvMode.OPTIONS:
			self.action_space = np.array([0, 1])
		elif mode == EnvMode.ALL:
			self.action_space = np.array([0, 1, 2, 3, 4, 5])

		self.mode = mode
		
		self.observation_space = np.zeros(np.sum(self.occupancy == 0))
		self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]

		# Random number generator
		self.rng = np.random.RandomState()

		self.tostate = {}
		statenum = 0
		for i in range(13):
			for j in range(13):
				if self.occupancy[i,j] == 0:
					self.tostate[(i,j)] = statenum
					statenum += 1
		self.tocell = {v:k for k, v in self.tostate.items()}

		if goal_pos:
			self.goal = goal_pos
		else:
			self.goal = 62 # East doorway
		if start_pos:
			self.init_states = [start_pos]
		else:
			self.init_states = list(range(self.observation_space.shape[0]))
			self.init_states.remove(self.goal)
		#self.init_states = list(range(np.sum(self.occupancy == 0)))
		self._init_option_map()
			
	def render(self, show_goal=True):
		current_grid = np.array(self.occupancy)
		current_grid[self.current_cell[0], self.current_cell[1]] = -3
		if show_goal:
			goal_cell = self.tocell[self.goal]
			current_grid[goal_cell[0], goal_cell[1]] = 3
		
		return current_grid

	def reset(self):
		self.steps = 0
		state = self.rng.choice(self.init_states)
		self.current_cell = self.tocell[state]
		return state, self.steps

	def check_available_cells(self, cell):
		available_cells = []

		for action in range(4):
			next_cell = tuple(cell + self.directions[action])

			if not self.occupancy[next_cell]:
				available_cells.append(next_cell)

		return available_cells
		

	def step(self, action):
		'''
		Takes a step in the environment with 2/3 probability. And takes a step in the
		other directions with probability 1/3 with all of them being equally likely.
		'''
		if self.mode == EnvMode.OPTIONS:
			self.mode = EnvMode.TRIVIAL
			result = self.execute_option(action + 4)
			self.mode = EnvMode.OPTIONS
			return result

		if action == 4 or action ==5:
			return self.execute_option(action)
			
		next_cell = tuple(self.current_cell + self.directions[action])

		if not self.occupancy[next_cell]:

			if self.rng.uniform() < 1/3 and not self.is_hallway():
				available_cells = self.check_available_cells(self.current_cell)
				self.current_cell = available_cells[self.rng.randint(len(available_cells))]

			else:
				self.current_cell = next_cell

		state = self.tostate[self.current_cell]
		self.steps += 1

		# When goal is reached, it is done
		done = state == self.goal
		truncated = self.steps > self.max_steps

		if self.rendering:
			clear_output(True)
			plt.imshow(self.render(show_goal=True), cmap='Blues')
			plt.axis('off')
			plt.show()
		

		return state, float(done), done, truncated, self.steps
	
    	

	def get_room(self):
		if self.current_cell[0] < 6 and self.current_cell[1] < 6:
			return 1
		elif self.current_cell[0] < 7 and self.current_cell[1] >= 5:
			return 2
		elif self.current_cell[0] >= 6 and self.current_cell[1] < 6:
			return 3
		else:
			return 4
		
	def execute_option(self, option):

		switch_maps_2nd_action = False
		if (self.current_cell in [(3,6), (10, 6)] and option== 4) or (self.current_cell in [(6,2), (7, 9)] and option== 5):
			switch_maps_2nd_action = True

		reached_halway = False
		option_mapping = self.option_map[0] if option==4 else self.option_map[1]

		while not reached_halway:
			
			action = option_mapping[self.current_cell[0], self.current_cell[1]]

			new_state, reward, terminated, truncated, steps = self.step(action)		

			if switch_maps_2nd_action:
				option_mapping = self.option_map[0] if option==5 else self.option_map[1]
				switch_maps_2nd_action = False

			reached_halway = self.is_hallway()

			if terminated or truncated:
				break
		return new_state, reward, terminated, truncated, steps
	
	def is_hallway(self): 
		return self.current_cell in [(3,6), (6,2), (7,9), (10,6)]

	# def map_option_to_targets(self, option):
	# 	room = self.get_room()
	# 	if room == 1:
	# 		target = (3, 6) if option == 4 else (6, 2)
	# 		last_position = (3, 5) if option == 4 else (5, 2)
	# 		last_action = 3 if option == 4 else 1
	# 	elif room == 2:
	# 		target = (3, 6) if option == 4 else (7, 9)
	# 		last_position = (3, 7) if option == 4 else (6, 9)
	# 		last_action = 2 if option == 4 else 1
	# 	elif room == 3:
	# 		target = (10, 6) if option == 4 else (6, 2)
	# 		last_position = (10, 5) if option == 4 else (7, 2)
	# 		last_action = 3 if option == 4 else 0
	# 	else:
	# 		target = (10, 6) if option == 4 else (7, 9)
	# 		last_position = (10, 7) if option == 4 else (8, 9)
	# 		last_action = 2 if option == 4 else 0
	# 	return target, last_position, last_action

	def _init_option_map(self):
		move1 = self.occupancy.copy()
		move1[:,:] = -1

		move1[1:6,1:5] = 3
		move1[7:-1,1:5] = 3 
		move1[1:7,8:-1]  = 2
		move1[8:-1, 8:-1] = 2
		move1[1:3,5] = 1 
		move1[1:3,7]  = 1
		move1[4:6,5] = 0 
		move1[4:7,7]  = 0
		move1[7:10,5] = 1 
		move1[8:10,7]  = 1
		move1[11,5] = 0 
		move1[11,7]  = 0
		move1[10,5] = 3 
		move1[10,7]  = 2
		move1[3,5] = 3 
		move1[3,7]  = 2
		move1[3,6] = 3
		move1[6,2] = 0
		move1[7,9] = 0
		move1[10,6] = 3

		move2 = self.occupancy.copy()
		move2[:,:]  = -1

		move2[1:6,7:-1]  = 1 # 1
		move2[1:5,1:6] = 1
		move2[8:-1,1:6]  = 0  #0
		move2[9:-1, 7:-1] = 0
		move2[5, 3:6] = 2
		move2[7, 3:6] = 2
		move2[6, 10:12] = 2
		move2[8, 10:12] = 2
		move2[5, 1] = 3
		move2[7, 1] = 3
		move2[6, 7:9] = 3
		move2[8, 7:9] = 3
		move2[5,2] = 1 # 1
		move2[7,2] = 0 # 0
		move2[8,9] = 0 #0
		move2[6,9] = 1 #1
		move2[3,6] = 2
		move2[6,2] = 1
		move2[7,9] = 1
		move2[10,6] = 2

		self.option_map = np.concatenate([move1,move2]).reshape(2, 13, 13)
