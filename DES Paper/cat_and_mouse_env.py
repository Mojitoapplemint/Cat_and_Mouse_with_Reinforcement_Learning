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
    metadata = {"render_modes": ["human"], "render_fps":4}
    
    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(low= 1, high=5, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(6)
        
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
    
    def update_door(self, event):
        cat_reward = 0
        mouse_reward = 0
        
        for i in range(3):
            if i==event:
                self.doors[i] = 1
            else:
                self.doors[i] = 0
                mouse_reward -= 2
        
        for i in range(3, 6):
            if i==event:
                self.doors[i] = 1
            else:
                self.doors[i] = 0
                cat_reward -= 2
                
        return mouse_reward, cat_reward
        
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
        
        
    def reset(self, seed=None, options = None):
        """
        Reset the environment to its initial state.
        Returns:
            np.ndarray: The initial state of the doors.
            dict: A dictionary containing the initial positions of the cat and mouse.
        """
        self.cat_position = 4
        self.mouse_position = 2
        self.doors = np.ones(shape=(6,), dtype=np.int32) #[m1, m2, m3, c1, c2, c3]
        
        observation = np.array([self.mouse_position, self.cat_position])
        info = {"doors": self.doors}
        return observation, info
    
    
    '''
        1 = Enable = open
        0 = Disable = close
    '''
    def step(self, event):
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
        
        mouse_r1, cat_r1 = self.update_door(event)
        
        mouse_r2 = self.mouse_move()
        cat_r2 = self.cat_move()
        
        if self.cat_position == 3 and self.mouse_position == 3:
            terminated = True
            cat_r2 = -10
            mouse_r2 = -10
        
        info = {"doors":self.doors}

        return np.array([self.mouse_position, self.cat_position]), (mouse_r1, mouse_r2, cat_r1, cat_r2), terminated, False, info
        

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
        
