import gymnasium as gym
import numpy as np
import cat_and_mouse_env
import pandas as pd

EVENTS = ["m1", "m2", "m3", "c1", "c2", "c3"]
MOUSE_OBSERVABLE_EVENTS = ["m1", "m2", "m3", "c1", "c2"]
CAT_OBSERVABLE_EVENTS = ["m2", "m3", "c1", "c2", "c3"]

MOUSE_STATES = {
        (1,3):0,
        (1,4):1, (1,5):1,
        (2,3):2,
        (2,4):3, (2,5):3,
        (3,3):4,
        (3,4):5, (3,5):5
    }

CAT_STATES = {
        (1,3):0, (2,3):0,
        (1,4):1, (2,4):1, 
        (1,5):2, (2,5):2, 
        (3,3):3,
        (3,4):4, 
        (3,5):5
    }

FEASIBLE_EVENTS = {
    (1,3):("m2", "c3"),
    (1,4):("m2", "c1"),
    (1,5):("m2", "c2"),
    
    (2,3):("m1", "c3"),
    (2,4):("m1", "c1"),
    (2,5):("m1", "c2"),
    
    (3,3):("m3", "c3"),
    (3,4):("m3", "c1"),
    (3,5):("m3", "c2"),
}

def policy_num_to_binary_list(policy_num):
    """ 
    Takes policy number (0-7) and return the binary list corresponds to the number
        1 represents enabled and opened
        0 represents disabled and closed
    Ex)
        0 = [0, 0, 0]
        1 = [0, 0, 1]
        2 = [0, 1, 0]
        3 = [0, 1, 1]
        4 = [1, 0, 0]
        5 = [1, 0, 1]
        6 = [1, 1, 0]
        7 = [1, 1, 1]

    Args:
        policy_num (int): Column number that represents policy (0-7)
    Returns:
        list of (0,1): Binary list
    """
    binary = np.binary_repr(policy_num)
    while len(binary)<3:
        binary = "0"+binary

    binary_int = int(binary)
    binary_list = []
    for _ in range(3):
        binary_list.insert(0, binary_int%10)
        binary_int = binary_int//10
    return binary_list

def local_policy_to_policy_num(local_policy, is_mouse):
    binary_list = []
    for door in EVENTS:
        if door in local_policy:
            binary_list.append(1)
        else:
            binary_list.append(0)
            
    if is_mouse:
        binary_list =  binary_list[:3]
    else:
        binary_list = binary_list[3:]
    
    binary = 100*binary_list[0]+10*binary_list[1]+binary_list[2]
    return int(str(binary), 2)

# [m1, m2, m3, c1, c2, c3]
# [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def init_mouse_eta():
                   #[m1 , m2 , m3,  c2 , c3]
    init_mouse_eta=[[0.0, 0.5, 0.0, 0.0, 0.5],
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                    [0.5, 0.0, 0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5, 0.0, 0.5],
                    [0.0, 0.0, 0.5, 0.5, 0.0]]
    return init_mouse_eta

def init_cat_eta():
                  # [m2,  m3,  c1,  c2,  c3]
    init_cat_eta = [[0.5, 0.0, 0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.0, 0.5],
                    [0.0, 0.5, 0.5, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.5, 0.0]]
    return init_cat_eta

def mouse_observation_to_state(observation):
    return MOUSE_STATES.get(tuple(observation))

def cat_observation_to_state(observation):
    return CAT_STATES.get(tuple(observation))

def get_mouse_policy(q_mouse, curr_state, observation, epsilon)->list:
    if np.random.random()>epsilon:
        # Exploration
        policy_num = 1-np.argmax(q_mouse[curr_state])
    else:
        # Exploitation
        policy_num = np.argmax(q_mouse[curr_state])
        
    # if np.random.random()>epsilon:
    #     # Exploration
    #     policy_num = 1
    # else:
    #     # Exploitation
    #     policy_num = 0
    
    # print(FEASIBLE_EVENTS.get(tuple(observation)))
    
    if policy_num == 1:
        policy = [FEASIBLE_EVENTS.get(tuple(observation))[0]]+["c1", "c2", "c3"]
    else:
        policy = ["c1", "c2", "c3"]
    
    return policy, policy_num

def get_cat_policy(q_cat, curr_state, observation,  epsilon)->list:
    
    if np.random.random()>epsilon:
        # Exploration
        policy_num = 1-np.argmax(q_cat[curr_state])
    else:
        # Exploitation
        policy_num = np.argmax(q_cat[curr_state])
        
    # if np.random.random()>epsilon:
    #     # Exploration
    #     policy_num = 1
    # else:
    #     # Exploitation
    #     policy_num = 0

    if policy_num == 1:
        policy = ["m1", "m2", "m3"]+[FEASIBLE_EVENTS.get(tuple(observation))[1]]
    else:
        policy = ["m1", "m2", "m3"]
    
    return policy, policy_num


def get_net_policy(cat_policy, mouse_policy):
    net_policy = []
    for i in cat_policy:
        if i in mouse_policy:
            net_policy.append(i)
    return net_policy

# DM = Dummy -1 (Since eta value is always positive, )
# [m1, m2, m3, c1, c2, c3] net
# [DM, m2, m3, c1, c2, c3] cat
# [m1, m2, m3, DM, c2, c3] mouse

def get_event(net_policy, mouse_state, cat_state, eta_mouse, eta_cat):
    eta_mouse_state = eta_mouse[mouse_state]
    dummy_eta_mouse = np.concatenate((eta_mouse_state[0:3], [-1], eta_mouse_state[3:]))
    
    eta_cat_state = eta_cat[cat_state]
    dummy_eta_cat = np.concatenate(([-1], eta_cat_state))

    max_eta = 0    
    event = None
    for curr_event in net_policy:
        curr_event_num = EVENTS.index(curr_event)
        curr_eta = max(dummy_eta_mouse[curr_event_num], dummy_eta_cat[curr_event_num])
        
        if max_eta <= curr_eta:
            max_eta = curr_eta
            event = curr_event
    return event

def get_disabled_event(net_policy, observation):
    feasible_events = FEASIBLE_EVENTS.get(tuple(observation))

    disabled = []
    for event in EVENTS:
        if (event not in net_policy) and (event in feasible_events):
            disabled.append(event)
            
    return disabled            
    

def update_t(t_table, old_state, new_state, event, r2, alpha, gamma, is_mouse):
    if is_mouse and event not in MOUSE_OBSERVABLE_EVENTS:
        return
    elif not is_mouse and event not in CAT_OBSERVABLE_EVENTS:
        return
    
    if is_mouse:
        event_num = MOUSE_OBSERVABLE_EVENTS.index(event)
    else:
        event_num = CAT_OBSERVABLE_EVENTS.index(event)
        
    t_table[old_state, event_num] = t_table[old_state, event_num]+alpha*(r2+gamma*max(t_table[new_state])-t_table[old_state, event_num])

def update_R1(R1_table, state, local_policy_num, r1, beta):
    R1_table[state, local_policy_num] = R1_table[state, local_policy_num]+beta*(r1-R1_table[state, local_policy_num])
    

def update_eta(eta_table, state, local_policy:list, event, delta, is_mouse):

    observable_policy = local_policy.copy()
    if is_mouse:
        observable_policy.remove("c1")
    else:
        observable_policy.remove("m1")
    
    for event_prime in observable_policy:
        event_prime_index = observable_policy.index(event_prime)
        if event_prime == event:
            eta_table[state, event_prime_index] = eta_table[state, event_prime_index]+delta*(np.sum(eta_table[state])-eta_table[state, event_prime_index])
        else:
            eta_table[state, event_prime_index] = (1-delta)*eta_table[state, event_prime_index]

def update_Q(q_table, T_table, R1_table, eta_table, state, policy_num):
    eta_sum = np.sum(eta_table[state])
    # if eta_sum == 0:
        # print(eta_table)
        # print(R1_table[state, policy_num]+np.dot(eta_table[state], T_table[state])/eta_sum)
        
    q_table[state, policy_num] = R1_table[state, policy_num]+np.dot(eta_table[state], T_table[state])/eta_sum
    

# Top Level Code
env = gym.make("CatAndMouse-v0", render_mode = "human")

q_mouse = np.zeros(shape=(6,2)) # 0: Disable controllable & feasible event, 1: Enable controllable & feasible event
q_cat = np.zeros(shape=(6,2)) # 0: Disable controllable & feasible event, 1: Enable controllable & feasible event

# (State s_i, 5 observable events) Each SuperVisor can observe 5 events
t_mouse = np.zeros(shape=(6,5)) 
t_cat = np.zeros(shape=(6,5))

# (State s_i, all possible control policy) 2^3 = 8 control policies
R1_mouse = np.zeros(shape=(6,2))
R1_cat = np.zeros(shape=(6,2))

# (State s_i, 5 observable events) Each SuperVisor can observe 5 events
eta_mouse = np.array(init_mouse_eta())
eta_cat= np.array(init_cat_eta())

epoch= 100
alpha = 0.1
beta = 0.1
gamma = 0.9
delta = 0.1
epsilon = 0.9

train_count=[0,0,0,0,0,0]

for episode in range(epoch):
    if (episode%100==0):
        print(str(100*episode/epoch)+"%","done" , end="\r")
        
    observation, info = env.reset()
    terminated = False
    
    print(observation)
    print()
    
    new_mouse_state = mouse_observation_to_state(observation)
    new_cat_state = cat_observation_to_state(observation)
    
    count = 0
    
    while (not terminated):
        
        # Get control policies for each SuperVisor
        mouse_policy, mouse_policy_num = get_mouse_policy(q_mouse, new_mouse_state, observation, epsilon)
        cat_policy, cat_policy_num = get_cat_policy(q_cat, new_cat_state, observation, epsilon)

        # Get net policy
        net_policy = get_net_policy(cat_policy, mouse_policy)
        
        # If net_policy is empty, then continue
        if len(net_policy)==0:
            # print("Net policy is empty")
            count +=1
            continue
        
        # Get action from net policy
        event = get_event(net_policy, new_mouse_state, new_cat_state, eta_mouse, eta_cat)
        
        train_count[EVENTS.index(event)] +=1
        
        # Get Feasible event at current state that is not included in net_policy
        disabled = get_disabled_event(net_policy, observation)
        
        print(observation)
        print(mouse_policy)
        print(cat_policy)
        print(net_policy)
        print(event)
        print("disabled",disabled)
        
        # Send Action to DES
        observation, reward, terminated, _, info = env.step((event, disabled))
        
        mouse_r1, mouse_r2, cat_r1, cat_r2 = reward
        
        # Storing old states
        old_mouse_state = new_mouse_state
        old_cat_state = new_cat_state
        
        # Getting new states
        new_mouse_state = mouse_observation_to_state(observation)
        new_cat_state = cat_observation_to_state(observation)
        
        # Updating T
        update_t(t_mouse, old_mouse_state, new_mouse_state, event, mouse_r2, alpha, gamma, True)
        update_t(t_cat, old_cat_state, new_cat_state, event, cat_r2, alpha, gamma, False)
        
        # Updating R1   
        update_R1(R1_mouse, old_mouse_state, mouse_policy_num, mouse_r1, beta)
        update_R1(R1_cat, old_cat_state, cat_policy_num, cat_r1, beta)
        
        # Updating eta
        update_eta(eta_mouse, old_mouse_state, mouse_policy, event, delta, True)
        update_eta(eta_cat, old_cat_state, cat_policy, event, delta, False)
        
        # Updating Q
        update_Q(q_mouse, t_mouse, R1_mouse, eta_mouse, old_mouse_state, mouse_policy_num)
        update_Q(q_cat, t_cat, R1_cat, eta_cat, old_cat_state, cat_policy_num)
        
        # print(q_mouse)
        # print(q_cat)
        # print()
        
        if count == 20:
            terminated = True
        count +=1

for i in range(6):
    print(f"{EVENTS[i]}: {train_count[i]}")

df_mouse = pd.DataFrame(q_mouse)
df_cat = pd.DataFrame(q_cat)
df_eta_mouse = pd.DataFrame(eta_mouse)
df_eta_cat = pd.DataFrame(eta_cat)

df_mouse.to_csv("q_mouse.csv")
df_cat.to_csv("q_cat.csv")
df_eta_mouse.to_csv("eta_mouse.csv")
df_eta_cat.to_csv("eta_cat.csv")