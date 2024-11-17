import gymnasium as gym
import numpy as np
import cat_and_mouse_env

states = [
        (1,3),
        (1,4),
        (1,5),
        (2,3),
        (2,4),
        (2,5),
        (3,3),
        (3,4),
        (3,5)
    ]

def row_to_state(row_num):
    return states[row_num]

def state_to_row(state):
    return states.index(state)

def get_mouse_policy(curr_state)->list:
    pass

def get_cat_policy(curr_state)->list:
    pass

def get_net_policy(cat_policy, mouse_policy):
    net_policy = []
    for i in range(6):
        if cat_policy[i]==1 and mouse_policy[i]==1:
            net_policy.append(i)
    return net_policy

def get_action(net_policy):
    pass

env = gym.make("CatAndMouse-v0")
q_mouse = np.zeros(shape=(9,6))
q_cat = np.zeros(shape=(9,6))

epoch=1000000
learning_rate=0.9
discount_factor = 0.9
epsilon = 0.9

for episode in range(epoch):
    if (episode%1000==0):
        print(str(100*episode/epoch)+"%","done" , end="\r")

    observation, info = env.reset()
    terminated = False
    
    new_state = row_to_state(observation)
    
    count = 0
    
    while (not terminated):
        
        cat_policy = get_cat_policy(new_state)
        mouse_policy = get_mouse_policy(new_state)
        
        net_policy = get_net_policy(cat_policy, mouse_policy)
        
        if len(net_policy)==0:
            continue
        
        action = get_action(net_policy)
        
        observation, reward, terminated, info = env.step(action)
        
        old_state = new_state
        
        new_state = observation
        
        
        
        if count == 20:
            terminated = True
        count +=1
        