import gymnasium as gym
import numpy as np
import cat_and_mouse_env
import pandas as pd

EVENTS = ["m1", "m2", "m3", "c1", "c2", "c3"]
MOUSE_OBSERVABLE_EVENTS = ["m1", "m2", "m3", "c2", "c3"]
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
    """
    Converts a local policy to a policy number.
    This function takes a local policy and converts it into a binary number
    based on the presence of certain events (doors). The binary number is then
    converted to a decimal integer.
    Args:
        local_policy (list): A list of events, denoted as a subsequence of [m1, m2, m3, c1, c2, c3].
        is_mouse (bool): A boolean indicating whether the policy is for a mouse (True) or not (False).
    Returns:
        int: The policy number represented as a decimal integer.
    """
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


def init_mouse_eta():
    """
    Gives the initial eta table for mouse
    """
                   #[m1 , m2 , m3 , c2 , c3]
    init_mouse_eta=[[0.0, 0.5, 0.0, 0.0, 0.5],
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                    [0.5, 0.0, 0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5, 0.0, 0.5],
                    [0.0, 0.0, 0.5, 0.5, 0.0]]
    return init_mouse_eta


def init_cat_eta():
    """
    Gives the initial eta table for cat
    """
                  # [m2,  m3,  c1,  c2,  c3]
    init_cat_eta = [[0.5, 0.0, 0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.0, 0.5],
                    [0.0, 0.5, 0.5, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.5, 0.0]]
    return init_cat_eta


def mouse_observation_to_state(observation):
    """
    Converts [mouse position, cat position] to local state number of mouse
    Args:
        observation [int:[1-3], int:[3,5]]: list of [mouse position, cat position]

    Returns:
        int: int value representing the mouse state
    """
    return MOUSE_STATES.get(tuple(observation))


def cat_observation_to_state(observation):
    """
    Converts [mouse position, cat position] to local state number of cat
    Args:
        observation [int:[1-3], int:[3,5]]: list of [mouse position, cat position]

    Returns:
        int: int value representing the cat state
    """
    return CAT_STATES.get(tuple(observation))


def get_mouse_policy(q_mouse, curr_state, observation, epsilon)->list:
    """
    Determines the policy for the mouse based on the Q-values and the current state.
    Args:
        q_mouse (np.ndarray): The Q-table for the mouse.
        curr_state (int): The current state of the mouse.
        observation (list): The current observation of the environment: [cat position, mouse position].
        epsilon (float): The exploration-exploitation trade-off parameter.
    Returns:
        policy (list): The local policy for the mouse
        policy_num (int): Return 1 if the policy is to enable the feasible event, otherwise 0; represents the column number of the Q-table.
    """

    if np.random.random()>epsilon:
        # Exploration
        policy_num = 1-np.argmax(q_mouse[curr_state])
    else:
        # Exploitation
        policy_num = np.argmax(q_mouse[curr_state])
    
    if policy_num == 1:
        policy = [FEASIBLE_EVENTS.get(tuple(observation))[0]]+["c1", "c2", "c3"]
    else:
        policy = ["c1", "c2", "c3"]
    
    return policy, policy_num


def get_cat_policy(q_cat, curr_state, observation,  epsilon)->list:
    """
    Determines the policy for the cat based on the Q-values and the current state.
    
    Args:
        q_cat (np.ndarray): The Q-table for the cat.
        curr_state (int): The current state of the cat.
        observation (list): The current observation of the environment: [cat position, mouse position].
        epsilon (float): The exploration-exploitation trade-off parameter.
    Returns:
        policy (list): The local policy for the cat
        policy_num (int): Return 1 if the policy is to enable the feasible event, otherwise 0; represents the column number of the Q-table.
    """
    
    if np.random.random()>epsilon:
        # Exploration
        policy_num = 1-np.argmax(q_cat[curr_state])
    else:
        # Exploitation
        policy_num = np.argmax(q_cat[curr_state])
    
    if policy_num == 1:
        policy = ["m1", "m2", "m3"]+[FEASIBLE_EVENTS.get(tuple(observation))[1]]
    else:
        policy = ["m1", "m2", "m3"]
    
    return policy, policy_num


def get_net_policy(cat_policy, mouse_policy):
    """
    Get the net policy by taking the intersection of the cat and mouse policies.
    Args:
        cat_policy (list): A list of policies for the cat.
        mouse_policy (list): A list of policies for the mouse.
    Returns:
        list: A list containing the common policies between cat_policy and mouse_policy. Refer to Equation 13 in the paper
    """
    
    net_policy = []
    for i in cat_policy:
        if i in mouse_policy:
            net_policy.append(i)
    return net_policy


def get_event(net_policy, mouse_state, cat_state, eta_mouse, eta_cat):
    """
    Determines one single event that has the highest local eta value 
    Args:
        net_policy (list): A list of events representing the global policy.
        mouse_state (int): The current state in perspective of the mouse supervisor.
        cat_state (int): The current state in perspective of the cat supervisor.
        eta_mouse (np.ndarray): A 2D array stores each the local eta values of mouse for (state, policy) combination.
        eta_cat (np.ndarray): A 2D array stores each the local eta values of cat for (state, policy) combination.
    Returns:
        event (str): The event with the highest eta value based on the given policy and states.
    """
    
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
    """
    Determine the disabled events based on the net policy and current observation(cat position, mouse position).
    Those events are basically feasible events of the current state(observation) that is not included in the net policy
    Args:
        net_policy (list): a net policy derived from local policies.
        observation (list): The current state observation(cat position, mouse position).
    Returns:
        list: A list of events that are disabled (not allowed by the network policy but feasible in the current observation).
    """
    
    feasible_events = FEASIBLE_EVENTS.get(tuple(observation))

    disabled = []
    for event in EVENTS:
        if (event not in net_policy) and (event in feasible_events):
            disabled.append(event)
            
    return disabled            
    

def update_t(t_table, old_state, new_state, event, r2, alpha, gamma, is_mouse):
    """
    Updates the T value for (old_state, event) pair based on the Equation 21 in the paper
    Args:
        t_table (numpy.ndarray): The t table to be updated.
        old_state (int): The previous state before the trainsition (denoted as s_i).
        new_state (int): The state after the trainsition (denoted as s'_i).
        event (str): The event that was occured.
        r2 (float): The reward received as the state transitioned.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        is_mouse (bool): Flag indicating whether the local agent is a mouse or not.
    """
    
    if is_mouse and event not in MOUSE_OBSERVABLE_EVENTS:
        return
    elif not is_mouse and event not in CAT_OBSERVABLE_EVENTS:
        return
    
    if is_mouse:
        event_num = MOUSE_OBSERVABLE_EVENTS.index(event)
    else:
        event_num = CAT_OBSERVABLE_EVENTS.index(event)
        
    t_table[old_state, event_num] = t_table[old_state, event_num]+alpha*(r2+gamma*max(t_table[new_state])-t_table[old_state, event_num])

def update_R1(R1_table, old_state, local_policy_num, r1, beta):
    """
    Update the R1 value for (old_state, event) pair based on the Equation 22 in the paper.
    Args:
        R1_table (numpy.ndarray): The table containing R1 values.
        state (int): The state before the transition.
        local_policy_num (int): The integer[0-5] representing the local policy.
        r1 (float): The reward value from the evaluation for the control pattern(local policy)
        beta (float): The learning rate for the update.

    """    
    R1_table[old_state, local_policy_num] = R1_table[old_state, local_policy_num]+beta*(r1-R1_table[old_state, local_policy_num])
    

def update_eta_mouse(eta_mouse, old_state, event, delta):
    """
    Update the eta values of (old_state, event) pair for the mouse, refer to Equation 23 in the paper.
    
    Note that eta values only for feasible events are updated, meaning that if the local supervisor cannot observe the event, the eta value is not updated.
    Parameters:
        eta_mouse (np.ndarray): A 2D numpy array representing the eta values for the mouse; row and column represents states and event, respectively.
        state (int): The local state of the mouse before the transition.
        event (int): The event that has occurred.
        delta (float): The learning rate or step size for updating eta values.
    """
    
    if event not in MOUSE_OBSERVABLE_EVENTS:
        return
    
    event_index = MOUSE_OBSERVABLE_EVENTS.index(event)
    
    new_event_eta = eta_mouse[old_state, event_index]+delta*(np.sum(eta_mouse[old_state])-eta_mouse[old_state, event_index])
    
    eta_mouse[old_state] = eta_mouse[old_state]*(1-delta)
    
    eta_mouse[old_state, event_index] = new_event_eta
    
    
def update_eta_cat(eta_cat, state, event, delta):
    """
    Update the eta values of (old_state, event) pair for the cat, refer to Equation 23 in the paper.
    
    Note that eta values only for feasible events are updated, meaning that if the local supervisor cannot observe the event, the eta value is not updated.
    Parameters:
        eta_mouse (np.ndarray): A 2D numpy array representing the eta values for the cat; row and column represents states and event, respectively.
        state (int): The local state of the cat before the transition.
        event (int): The event that has occurred.
        delta (float): The learning rate or step size for updating eta values.
    """ 
    if event not in CAT_OBSERVABLE_EVENTS:
        return
    
    event_index = CAT_OBSERVABLE_EVENTS.index(event)
    
    new_event_eta = eta_cat[state, event_index]+delta*(np.sum(eta_cat[state])-eta_cat[state, event_index])
    
    eta_cat[state] = eta_cat[state]*(1-delta)
    
    eta_cat[state, event_index] = new_event_eta
    

def update_Q(q_table, T_table, R1_table, eta_table, state, policy_num):
    """
    Updates the Q-table for a given state and control policy, refer to Equation 24 in the paper.
    Parameters:
        q_table (np.ndarray): The Q-table to be updated.
        T_table (np.ndarray): T table.
        R1_table (np.ndarray): R1 table.
        eta_table (np.ndarray): eta table.
        state (int): The state before transition.
        policy_num (int): The policy number.
    """
    
    eta_sum = np.sum(eta_table[state])
        
    q_table[state, policy_num] = R1_table[state, policy_num]+np.dot(eta_table[state], T_table[state])/eta_sum
    

#--------------Top Level Code------------------#
env = gym.make("CatAndMouse-v0", render_mode = "human")

q_mouse = np.zeros(shape=(6,2)) # 0: Disable controllable & feasible event, 1: Enable controllable & feasible event
q_cat = np.zeros(shape=(6,2)) # 0: Disable controllable & feasible event, 1: Enable controllable & feasible event

# (State s_i, 5 observable events) Each SuperVisor can observe 5 events
t_mouse = np.zeros(shape=(6,5)) 
t_cat = np.zeros(shape=(6,5))

# (State s_i, all feasible control policy)
R1_mouse = np.zeros(shape=(6,2))
R1_cat = np.zeros(shape=(6,2))

# (State s_i, 5 observable events) Each SuperVisor can observe 5 events
eta_mouse = np.array(init_mouse_eta())
eta_cat= np.array(init_cat_eta())

epoch= 1000
alpha = 0.1
beta = 0.1
gamma = 0.9
delta = 0.1
epsilon = 0.9

train_count=[0,0,0,0,0,0]

for episode in range(epoch):
    if (episode%100==0):
        print(str(100*episode/epoch)+"%","done" , end="\r")
    
    # print("New Episode")
    
    observation, info = env.reset()
    terminated = False
    
    # print(observation)
    # print()
    
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
        
        # print(event)
        
        train_count[EVENTS.index(event)] +=1
        
        # Get Feasible event at current state that is not included in net_policy
        disabled = get_disabled_event(net_policy, observation)
        
        # print(observation)
        # print(mouse_policy)
        # print(cat_policy)
        # print(net_policy)
        # print("Enabled", event, "/ disabled",disabled)
        
        # Send Action to DES
        observation, reward, terminated, _, info = env.step((event, disabled))
        
        mouse_r1, mouse_r2, cat_r1, cat_r2 = reward
        
        # print(mouse_r1, mouse_r2, cat_r1, cat_r2)
        
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
        update_eta_mouse(eta_mouse, old_mouse_state, event, delta)
        update_eta_cat(eta_cat, old_cat_state, event, delta)
        
        # Updating Q
        update_Q(q_mouse, t_mouse, R1_mouse, eta_mouse, old_mouse_state, mouse_policy_num)
        update_Q(q_cat, t_cat, R1_cat, eta_cat, old_cat_state, cat_policy_num)
        
        if count == 20:
            terminated = True
        count +=1

for i in range(6):
    print(f"{EVENTS[i]}: {train_count[i]}")

df_mouse = pd.DataFrame(q_mouse)
df_cat = pd.DataFrame(q_cat)
df_mouse.to_csv("q_mouse.csv")
df_cat.to_csv("q_cat.csv")

df_eta_mouse = pd.DataFrame(eta_mouse)
df_eta_cat = pd.DataFrame(eta_cat)
df_eta_mouse.to_csv("eta_mouse.csv")
df_eta_cat.to_csv("eta_cat.csv")

df_t_mouse = pd.DataFrame(t_mouse)
df_t_cat = pd.DataFrame(t_cat)
df_t_mouse.to_csv("t_mouse.csv")
df_t_cat.to_csv("t_cat.csv")
