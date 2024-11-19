import gymnasium as gym
import numpy as np
from gymnasium import spaces

gym.register(
    id="CatAndMouse-v0",
    entry_point="cat_and_mouse_env:CatAndMouseEnv"
)

class CatAndMouseEnv(gym.Env):
    """
    Initialize the CatAndMouseEnv environment.
    Args:
        render_mode (str, optional): The mode to render the environment. Defaults to None.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.int32)
        self.action_space = spaces.Discrete(6)
        
        assert render_mode is None or render_mode in self.metadata['render.modes']
        self.render_mode = render_mode
    
    def update_door(self, action):
        cat_reward = 0
        mouse_reward = 0
        
        for i in range(3):
            if self.close_door(i, action[i]):
                mouse_reward -= 2
        
        for i in range(3, 6):
            if self.close_door(i, action[i]):
                cat_reward -= 2
                
        return cat_reward, mouse_reward
        
    
    def close_door(self, door_id, action):
        """
        Close or open a door based on the action.
        Args:
            door_id (int): The ID of the door to be closed or opened.
            action (int): The action to be performed (0 to close, 1 to open).
        Returns:
            bool: True if the door was closed, False otherwise.
        """
        
        if self.doors[door_id] == 1 and action == 0:
            self.doors[door_id] = 0
            return True
        
        self.doors[door_id] = 1
        return False
        
    def cat_move(self):
        
        if self.cat_position == 3:
            door_num = 5 # c3
            
        if self.cat_position == 4:
            door_num = 3 # c1
        
        if self.cat_position == 5:
            door_num = 4 # c2
        
        if self.doors[door_num] == 1:        
            self.cat_position = self.cat_position + 1
            if self.cat_position == 6:
                self.cat_position = 3
            return 1
        return 0
            
    
    def mouse_move(self):
        
        if self.mouse_position == 1:
            door_num = 1 # m2
            
        if self.mouse_position == 2:
            door_num = 0 # m1
        
        if self.mouse_position == 3:
            door_num = 2 # m3

        if self.doors[door_num] == 1:        
            self.mouse_position = self.mouse_position - 1
            if self.mouse_position == 0:
                self.mouse_position = 3
            return 1
        return 0
        
        
    def reset(self):
        """
        Reset the environment to its initial state.
        Returns:
            np.ndarray: The initial state of the doors.
            dict: A dictionary containing the initial positions of the cat and mouse.
        """
        self.cat_position = 4
        self.mouse_position = 2
        self.doors = np.ones(shape=(6,), dtype=np.int32) #[m1, m2, m3, c1, c2, c3]
        
        observation = (self.cat_position, self.mouse_position)
        info = {"doors": self.doors}
        return observation, info
    
    
    '''
        1 = Enable = open
        0 = Disable = close
    '''
    def step(self, action):
        """
        Take a step in the environment with the given actions for the cat and mouse.
        Args:
            cat_action (dict): The action for the cat.
            mouse_action (dict): The action for the mouse.
        Returns:
            tuple: A tuple containing the observations for the cat and mouse.
            tuple: A tuple containing the rewards for the cat and mouse.
            bool: Whether the episode has terminated.
            dict: A dictionary containing the positions of the cat and mouse.
        """
        terminated = False
        
        cat_reward, mouse_reward = self.update_door(action)
        
        cat_reward += self.cat_move(self)
        mouse_reward += self.mouse_move(self)
        
        if self.cat_position == 3 and self.mouse_position == 3:
            terminated = True
            cat_reward = -10
            mouse_reward = -10
        
        info = {"doors":self.doors}

        return (self.cat_position, self.mouse_position), (cat_reward, mouse_reward), terminated, info
        
        
    def render(self):
        """
        Render the environment.
        """
        cat = self.cat_position-1
        mouse = self.mouse_position-1
        if self.cat_position == 3:
            cat=5

        grid = [" " for _ in range(6)]
        grid[cat] = "C"
        grid[mouse] = "M"
        b = []
        print(f"\
    _________________________\n\
    |   1   |       |   4   |\n\
    |   {b[0]}   |   3   |   {b[3]}   |\n\
    ---------   {b[2]}   ---------\n\
    |   2   |   {b[5]}   |   5   |\n\
    |   {b[1]}   |       |   {b[4]}   |")
        
