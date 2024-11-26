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
    
    EVENTS = ["m1", "m2", "m3", "c1", "c2", "c3"]
    
    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(low= 1, high=5, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(6)
        
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
    
    def update_door(self, event, disabled):
        #Change Disbled 
        cat_reward = 0
        mouse_reward = 0
        
        # Enabling event
        self.doors[self.EVENTS.index(event)] = 1
        
        # Disabling event(s)
        for d_event in disabled:
            d_event_num = self.EVENTS.index(d_event)
            if self.doors[d_event_num] == 1:
                self.doors[d_event_num] = 0
                if d_event_num<3:
                    mouse_reward-=2
                else:
                    cat_reward-=2
                
        return mouse_reward, cat_reward
        
    def cat_move(self, event):
        
        if self.cat_position == 3:
            required_event = "c3"
            
        if self.cat_position == 4:
            required_event = "c1"
        
        if self.cat_position == 5:
            required_event = "c2"
        
        if required_event == event:        
            self.cat_position = self.cat_position + 1
            if self.cat_position == 6:
                self.cat_position = 3
            return 1
        return 0
            
    
    def mouse_move(self, event):
        
        if self.mouse_position == 1:
            required_event = "m2"
            
        if self.mouse_position == 2:
            required_event = "m1"
        
        if self.mouse_position == 3:
            required_event = "m3"

        if required_event == event:        
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
    def step(self, events):
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
        
        event, disabled = events
        
        mouse_r1, cat_r1 = self.update_door(event, disabled)
        
        #Let it move only when the observer observe events
        mouse_r2 = self.mouse_move(event)
        cat_r2 = self.cat_move(event)
        
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
        
