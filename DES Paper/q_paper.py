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

def get_mouse_policy(curr_state, epsilon)->list:
    mouse_policy = []
    if np.random.random > epsilon:
        for i in range(3):
            if np.random.random() > 0.5:
                mouse_policy.append(1)
            else:
                mouse_policy.append(0)
    else:
        for i in range(3):
            if q_mouse[curr_state, 2*i] > q_mouse[curr_state, 2*i+1]:
                mouse_policy.append(1)
            else:
                mouse_policy.append(0)
    return mouse_policy+[1,1,1]

def get_cat_policy(curr_state)->list:
    cat_policy = []
    if np.random.random > epsilon:
        for i in range(3):
            if np.random.random() > 0.5:
                cat_policy.append(1)
            else:
                cat_policy.append(0)
    else:
        for i in range(3):
            if q_cat[curr_state, 2*i] > q_cat[curr_state, 2*i+1]:
                cat_policy.append(1)
            else:
                cat_policy.append(0)
    return [1,1,1]+cat_policy

#  원래는 net_policy를 구해서 action을 eta 값에 근거해서 선택해야하는데, eta 값이 SV에 따라 다르게 나올텐데 누구걸 써야하는 지 모르겠음
# 
# def get_net_policy(cat_policy, mouse_policy):
#     net_policy = []
#     for i in range(6):
#         if cat_policy[i]==1 and mouse_policy[i]==1:
#             net_policy.append(i)
#     return net_policy

# def get_action(net_policy):
#     pass

env = gym.make("CatAndMouse-v0")
q_mouse = np.zeros(shape=(6,6))   # Observation: 개수 줄여 / Action: [m1 ON, m1 OFF, m2 ON, m2 OFF, m3 ON, m3 OFF]
q_cat = np.zeros(shape=(6,6))     # Observation: 개수 줄여 / Action: [c1 ON, c1 OFF, c2 ON, c2 OFF, c3 ON, c3 OFF]

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
            count +=1
            continue
        
        action = get_action(net_policy)
        
        observation, reward, terminated, info = env.step(action)
        
        old_state = new_state
        
        new_state = row_to_state(observation)
        
        
        
        if count == 20:
            terminated = True
        count +=1
        