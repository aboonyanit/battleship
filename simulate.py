import random
import numpy as np
import math

def explore(A, N, Q, c, s):
    # N: visit counts for each state-action pair
    # Q: action value estimates for each state-action pair
    # c: exploration constant
    # s: current state
    # A: action space
    s_tup = tuple(map(tuple, s))
    numVisits = 0
    for a in A:
        s_a_pair = s_tup + tuple(a)
        if s_a_pair in N.keys():
            numVisits += N[s_a_pair]
    maxVal = -1*float("Inf")
    maxA = None
    for a in A:
        s_a_pair = s_tup + tuple(a)
        if s_a_pair in N.keys():
            if N[s_a_pair] == 0:
                val = float('inf')
            else:
                val = Q[s_a_pair] + c * math.sqrt(math.log(numVisits) / N[s_a_pair])
            if val > maxVal:
                maxVal = val
                maxA = a
    return maxA

#best action with actual probability of hit
def state_action_sim(s, ship_sizes, *args):
    r = rewards(s)
    counts = simulated_counts(s, ship_sizes)
    idx_hits = np.where(s == 2)
    for i in range(np.size(idx_hits[0])): #don't want to pick a location we've already hit
        counts[idx_hits[0][i],idx_hits[1][i]] = 0
    Ts = transition_probability(counts)
    a = np.array([])
    if len(args) > 0:
        a = args[0]
    else:
        i,j = np.unravel_index(Ts.argmax(), Ts.shape)
        a = np.array([i,j])
    n = 1 #number of trials
    p = Ts[a[0],a[1]] #probabiliy of success
    s[a[0],a[1]] = 1 + np.random.binomial(n, p) #hit = 2, miss = 1, not guessed = 0
    return s, r

# best action, 0.5 probability of hit
def state_action_sim_rand1(s, ship_sizes, *args):
    r = rewards(s)
    counts = simulated_counts(s, ship_sizes)
    idx_hits = np.where(s == 2)
    for i in range(np.size(idx_hits[0])): #don't want to pick a location we've already hit
        counts[idx_hits[0][i],idx_hits[1][i]] = 0
    Ts = transition_probability(counts)
    a = np.array([])
    if len(args) > 0:
        a = args[0]
    else:
        i,j = np.unravel_index(Ts.argmax(), Ts.shape)
        a = np.array([i,j])
    n = 1 #number of trials
    p = 0.5 #probabiliy of success
    s[a[0],a[1]] = 1 + np.random.binomial(n, p) #hit = 2, miss = 1, not guessed = 0
    return s, r

# random action, 0.5 probability of hit
def state_action_sim_rand2(s, ship_sizes, *args):
    r = rewards(s)
    a = np.array([])
    if len(args) > 0:
        a = args[0]
    else:
        idx_available_guesses = np.where(s == 0)
        idx = random.choice(range(np.size(idx_available_guesses[0])))
        a = [idx_available_guesses[0][idx], idx_available_guesses[1][idx]]
    n = 1 #number of trials
    p = 0.5
    s[a[0],a[1]] = 1 + np.random.binomial(n, p) #hit = 2, miss = 1, not guessed = 0
    return s, r

# random action, actual probability of hit
def state_action_sim_rand3(s, ship_sizes, *args):
    r = rewards(s)
    counts = simulated_counts(s, ship_sizes)
    idx_hits = np.where(s == 2)
    for i in range(np.size(idx_hits[0])): #don't want to pick a location we've already hit
        counts[idx_hits[0][i],idx_hits[1][i]] = 0
    Ts = transition_probability(counts)
    a = np.array([])
    if len(args) > 0:
        a = args[0]
    else:
        idx_available_guesses = np.where(s == 0)
        idx = random.choice(range(np.size(idx_available_guesses[0])))
        a = [idx_available_guesses[0][idx], idx_available_guesses[1][idx]]
    n = 1 #number of trials
    p = Ts[a[0],a[1]] #probabiliy of success
    s[a[0],a[1]] = 1 + np.random.binomial(n, p) #hit = 2, miss = 1, not guessed = 0
    return s, r

def rollout(s,ship_sizes,gamma, d):
    tot_num_hits = np.sum([ship_length*num_ships for ship_length, num_ships in ship_sizes.items()])
    if (d <= 0) or (np.sum(s == 0) == 0) or (np.sum(s == 2) == tot_num_hits):
        return 0
    s, r = state_action_sim_rand1(s,ship_sizes)
    return r + gamma*rollout(s,ship_sizes, gamma, d - 1)

def rewards(s):
    total_guessed = np.sum(s > 0)
    if total_guessed == 0:
        return 0
    else:
        return np.sum(s == 2)/np.sum(s > 0) #number of hits over number of guesses

def simulated_counts(s, ship_sizes):
    counts = np.zeros(np.shape(s))
    num_samples = 100
    for i in range(num_samples):
        available_space = (s != 1)
        for ship_length, num_ships in ship_sizes.items(): #will work better if ship sizes is organized in order of largests ship first
            for ship in range(num_ships):
                if ship_length < np.sum(available_space):
                    x1,y1,x2,y2 = try_to_place(ship_length,available_space, s)
                    if (x1 != x2) or (y1 != y2):
                        counts[y1:y2+1,x1:x2+1] += 1 #upper leftmost corner is x,y = 0,0
                        available_space[y1:y2+1,x1:x2+1] = 0
    return counts

# Notes: because there are so many combinations of ship locations, the probability of having a ship
# hit in the beginning of the game is low --> no hits till later in game (meaning we don't get any rewards if we don't go to full depth)
# (towards the end of the game this isn't always the case)
# maybe we can increase the probability a bit? depending on how many guesses are left.
# could also try making transition probabilities completely random if it runs too slow (regardless
# would be good to compare performance...)
def try_to_place(ship_length, available_space, s):
    hits_left = np.logical_and((s == 2),(available_space == 1))
    idx_hits_left = np.where(hits_left == 1)
    idx_available_space = np.where(available_space == 1)

    x1 = 0
    y1 = 0
    num_attempts = 0
    total_allowed_attempts = 20
    while num_attempts < total_allowed_attempts:
        if np.size(idx_hits_left) > 0 and (num_attempts < 0.5*total_allowed_attempts):
            idx = random.choice(range(np.size(idx_hits_left[0])))
            x1 = idx_hits_left[1][idx]
            y1 = idx_hits_left[0][idx]
        else:
            idx = random.choice(range(np.size(idx_available_space[0])))
            x1 = idx_available_space[1][idx]
            y1 = idx_available_space[0][idx]
        orientation = random.randrange(4)
        if orientation == 0: #go right 
            x2 = x1 + ship_length - 1
            y2 = y1
            if (x2 < s.shape[1]) and (np.sum(available_space[y1:y2+1,x1:x2+1]) == ship_length): 
                return x1,y1,x2,y2
        elif orientation == 1: #go down
            x2 = x1 
            y2 = y1 + ship_length - 1
            if (y2 < s.shape[0]) and (np.sum(available_space[y1:y2+1,x1:x2+1]) == ship_length): 
                return x1,y1,x2,y2
        elif orientation == 2: #go left 
            x2 = x1 - (ship_length - 1)
            y2 = y1 
            if (x2 >= 0) and (y2 < s.shape[0]) and (np.sum(available_space[y1:y2+1,x2:x1+1]) == ship_length): 
                return x2,y1,x1,y2
        else: #go up
            x2 = x1 
            y2 = y1 - (ship_length - 1)
            if (y2 >= 0) and (np.sum(available_space[y2:y1+1,x1:x2+1]) == ship_length): 
                return x1,y2,x2,y1
        num_attempts += 1
    return x1,y1,x1,y1

def transition_probability(counts):
    if np.sum(counts) == 0:
        return np.zeros(np.shape(counts))
    else:
        return counts/np.sum(counts)