import gymnasium as gym
import numpy as np
import cat_and_mouse_env

mouse_states = {
        (1,3):0,
        (1,4):1, (1,5):1,
        (2,3):2,
        (2,4):3, (2,5):3,
        (3,3):4,
        (3,4):5, (3,5):5
    }

cat_states = {
        (1,3):0, (2,3):0,
        (1,4):1, (2,4):1, 
        (1,5):2, (2,5):2, 
        (3,3):3,
        (3,4):4, 
        (3,5):5
    }

def mouse_observation_to_state(observation):
    return mouse_states.get(observation)

def get_mouse_policy(curr_state, exploration)->list:
    mouse_policy = []
    if exploration:
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

def cat_observation_to_state(observation):
    return cat_states.get(observation)

def get_cat_policy(curr_state, exploration)->list:
    cat_policy = []
    if exploration:
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

def get_net_policy(cat_policy, mouse_policy):
    net_policy = []
    for i in range(6):
        if cat_policy[i]==1 and mouse_policy[i]==1:
            net_policy.append(i)
    return net_policy

def get_action(net_policy):
    pass

env = gym.make("CatAndMouse-v0")
q_mouse = np.zeros(shape=(6,6))   # Observation: 개수 줄여 / Action: [m1 ON, m1 OFF, m2 ON, m2 OFF, m3 ON, m3 OFF]
q_cat = np.zeros(shape=(6,6))     # Observation: 개수 줄여 / Action: [c1 ON, c1 OFF, c2 ON, c2 OFF, c3 ON, c3 OFF]

epoch=1000000
alpha = 0.9
belta = 0.9
gamma = 0.9
delta = 0.9
epsilon = 0.9

for episode in range(epoch):
    if (episode%1000==0):
        print(str(100*episode/epoch)+"%","done" , end="\r")
        
    t_mouse = 0
    t_cat = 0
    R1_mouse = 0
    R1_cat = 0
    eta_mouse = 0
    eta_cat=0    
    
    observation, info = env.reset()
    terminated = False
    
    new_mouse_state = mouse_observation_to_state(observation)
    new_cat_state = cat_observation_to_state(observation)
    
    count = 0
    
    while (not terminated):
        exploration = np.random.random > epsilon
        
        # Get control policies for each SuperVisor
        cat_policy = get_cat_policy(new_cat_state, exploration)
        mouse_policy = get_mouse_policy(new_mouse_state, exploration)
        
        # Get net policy
        net_policy = get_net_policy(cat_policy, mouse_policy)
        
        # If net_policy is empty, then continue
        if len(net_policy)==0:
            print(f"cat_policy:{cat_policy}, mouse_policy:{mouse_policy}")
            count +=1
            continue
        
        # Get action from net policy
        action = get_action(net_policy)
        
        # Send Action to DES
        observation, cat_r1, cat_r2, mouse_r1, mouse_r2, terminated, info = env.step(action)
        
        # Storing old states
        old_cat_state = new_cat_state
        old_mouse_state = new_mouse_state
        
        # Getting new states
        new_cat_state = cat_observation_to_state(observation)
        new_mouse_state = mouse_observation_to_state(observation)
        
        # Updating T
        t_cat = t_cat+alpha[cat_r2+gamma*np.max(q_cat[new_cat_state])-t_cat]
        t_mouse = t_mouse+alpha[mouse_r2+gamma*np.max(q_mouse[new_mouse_state])-t_mouse]
        
        # Updating R1
        R1_cat = R1_cat + belta*(cat_r1-R1_cat)
        R1_mouse = R1_mouse + belta*(mouse_r1-R1_mouse)
        
        # Updating eta
        
        # Updating Q
        
        if count == 20:
            terminated = True
        count +=1
        