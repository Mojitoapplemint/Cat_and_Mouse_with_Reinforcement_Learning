import gymnasium as gym
import numpy as np
import cat_and_mouse_env

DOORS = ["m1", "m2", "m3", "c1", "c2", "c3"]

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
    return MOUSE_STATES.get(observation)

def get_mouse_policy(q_mouse, curr_state, epsilon)->list:
    if np.random()>epsilon:
        policy_num = np.argmax(q_mouse[curr_state])
    else:
        policy_num = np.random.randint(0,8)
    binary_list = policy_num_to_binary_list(policy_num)+[1,1,1]
    return [DOORS[i] for i in range(6) if binary_list[i]==1]


def cat_observation_to_state(observation):
    return CAT_STATES.get(observation)


def get_cat_policy(q_cat, curr_state, epsilon)->list:
    if np.random()>epsilon:
        policy_num = np.argmax(q_cat[curr_state])
    else:
        policy_num = np.random.randint(0,8)
    binary_list = [1,1,1]+policy_num_to_binary_list(policy_num)
    return [DOORS[i] for i in range(6) if binary_list[i]==1]


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

def get_event(net_policy, state, eta_mouse, eta_cat):
    eta_cat_state = eta_cat[state]
    dummy_eta_cat = [-1] + eta_cat_state
    
    eta_mouse_state = eta_mouse[state]
    dummy_eta_mouse = eta_mouse_state[0:3] + [-1] + eta_mouse_state[3:]
    
    max_eta = 0    
    event = None
    for curr_event in net_policy:
        curr_event_num = DOORS.index(curr_event)
        curr_eta = np.max(dummy_eta_mouse[curr_event_num], dummy_eta_cat[curr_event_num])
        if max_eta < curr_eta:
            max_eta = curr_eta
            event = curr_event
            
    return event


def update_t(t_table, old_state, new_state, action, r2, alpha, gamma):
    t_table[old_state, action] = t_table[old_state, action]+alpha*[r2+gamma*np.max(t_table[new_state])-t_table[old_state, action]]

def update_R1(R1_table, state, local_policy, r1, beta):
    R1_table[state, local_policy] = R1_table[state, local_policy]+beta*(r1-R1_table[state, local_policy])

def update_eta(eta_table, state, local_policy:list, event, delta, is_mouse):
    if is_mouse:
        observable_policy = local_policy.remove("c1")
    else:
        observable_policy = local_policy.remove("m1")
    
    for event_prime in observable_policy:
        event_prime_index = observable_policy.index(event_prime)
        if event_prime == event:
            eta_table[state, event_prime_index] = eta_table[state, event_prime_index]+delta*(np.sum(eta_table[state])-eta_table[state, event_prime_index])
        else:
            eta_table[state, event_prime_index] = (1-delta)*eta_table[state, event_prime_index]

def update_Q(q_table, T_table, R1_table, eta_table, state, policy):
    eta_sum = np.sum(eta_table[state])
    q_table[state, policy] = R1_table[state, policy]+np.dot(eta_table[state], T_table[state])/eta_sum
    

# Top Level Code
env = gym.make("CatAndMouse-v0")

q_mouse = np.zeros(shape=(6,8))   # Action: [m1 ON, m1 OFF, m2 ON, m2 OFF, m3 ON, m3 OFF]
q_cat = np.zeros(shape=(6,8))     # Action: [c1 ON, c1 OFF, c2 ON, c2 OFF, c3 ON, c3 OFF]

# (State s_i, 5 observable events) Each SuperVisor can observe 5 events
t_mouse = np.zeros(shape=(6,5)) 
t_cat = np.zeros(shape=(6,5))

# (State s_i, all possible control policy) 2^3 = 8 control policies
R1_mouse = np.zeros(shape=(6,8))
R1_cat = np.zeros(shape=(6,8))

# (State s_i, 5 observable events) Each SuperVisor can observe 5 events
eta_mouse = init_mouse_eta()
eta_cat= init_cat_eta()

epoch=1000000
alpha = 0.9
beta = 0.9
gamma = 0.9
delta = 0.9
epsilon = 0.9

for episode in range(epoch):
    if (episode%1000==0):
        print(str(100*episode/epoch)+"%","done" , end="\r")
        
    observation, info = env.reset()
    terminated = False
    
    new_mouse_state = mouse_observation_to_state(observation)
    new_cat_state = cat_observation_to_state(observation)
    
    count = 0
    
    while (not terminated):
        
        # Get control policies for each SuperVisor
        mouse_policy = get_mouse_policy(new_mouse_state, epsilon)
        cat_policy = get_cat_policy(new_cat_state, epsilon)
        
        # Get net policy
        net_policy = get_net_policy(cat_policy, mouse_policy)
        
        # If net_policy is empty, then continue
        if len(net_policy)==0:
            print(f"cat_policy:{cat_policy}, mouse_policy:{mouse_policy}")
            count +=1
            continue
        
        # Get action from net policy
        event = get_event(net_policy)
        
        # Send Action to DES
        observation, cat_r1, cat_r2, mouse_r1, mouse_r2, terminated, info = env.step(DOORS.index(event))
        
        # Storing old states
        old_cat_state = new_cat_state
        old_mouse_state = new_mouse_state
        
        # Getting new states
        new_cat_state = cat_observation_to_state(observation)
        new_mouse_state = mouse_observation_to_state(observation)
        
        # Updating T
        update_t(t_cat, old_cat_state, new_cat_state, event, cat_r2, alpha, gamma)
        update_t(t_mouse, old_mouse_state, new_mouse_state, event, mouse_r2, alpha, gamma)
        
        # Updating R1
        update_R1(R1_cat, old_cat_state, cat_policy, cat_r1, beta)
        update_R1(R1_mouse, old_mouse_state, mouse_policy, mouse_r1, beta)
        
        # Updating eta
        update_eta(eta_cat, old_cat_state, cat_policy, event, delta, False)
        update_eta(eta_mouse, old_mouse_state, mouse_policy, event, delta, True)
        
        # Updating Q
        update_Q(q_cat, t_cat, R1_cat, eta_cat, old_cat_state, cat_policy)
        update_Q(q_mouse, t_mouse, R1_mouse, eta_mouse, old_mouse_state, mouse_policy)
        
        if count == 20:
            terminated = True
        count +=1
        