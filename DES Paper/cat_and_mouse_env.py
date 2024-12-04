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
        """
        Updates the state of doors. Enables the door for parameter "event" and disables the doors for the events in the "disabled" list.
        
        If the door that each supervisor can control is disabled, then the supervisor gets a reward of -2.
        
        Args:
            event (str): The event to be enabled.
            disabled (list of str): List of events to be disabled.
        Returns:
            tuple: A tuple containing the mouse reward and the cat reward after updating the doors.
        """
        
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
        """
        Moves the cat to the next position based on whether the door for the next room is opened, or not.
        
        If the door is opened, the cat moves to the next room and supervisor get areward of 1, 
        otherwise the cat stays in the same room and supervisor get areward of 0.
        
        Args:
            event (str): The event that triggers the cat's movement. Expected values are "c1", "c2", or "c3".
        Returns:
            int: Returns 1 if the cat moves to the next position, otherwise returns 0.
        """
        
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
        """
        Moves the mouse to the next position based on whether the door for the next room is opened, or not.
        
        If the door is opened, the mouse moves to the next room and supervisor get areward of 1, 
        otherwise the mouse stays in the same room and supervisor get areward of 0.
        
        Args:
            event (str): The event that triggers the mouse movement. Expected values are "m1", "m2", or "m3".
        Returns:
            int: Returns 1 if the mouse position is updated successfully, otherwise returns 0.
        """
        
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
            list: [cat position, mouse position].
            dict: A dictionary containing the current status for each door, in a single list.
        """
        self.cat_position = 4
        self.mouse_position = 2
        self.doors = np.ones(shape=(6,), dtype=np.int32)
        
        observation = np.array([self.mouse_position, self.cat_position])
        info = {"doors": self.doors}
        return observation, info
    
    
    def step(self, events):
        """
        Takes information of doors that are planning to be opened or closed, updates the environment,
        and returns numerical rewards with the updated observation, (cat position, mouse position).
        
        Args:
            events (tuple): A tuple containing the event planning to be enabled, and list of events planning to be disbled.
        Returns:
            tuple: A tuple containing:
                - np.array: The current positions of the mouse and the cat.
                - tuple: Rewards for the mouse and the cat from the door update and their movements.
                - bool: A flag indicating if the episode has terminated.
                - bool: A flag indicating if the episode was truncated (always False in this case).
                - dict: Additional information about the environment (e.g., doors status).
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
        
