import random
import numpy as np

def state_action_sim(b,*args):
    """Compute the gradient of the loss with respect to theta."""
    r = rewards(b._boardHitMiss)
    counts = simulated_counts(b)
    idx_hits = np.where(b._boardHitMiss == 2)
    for i in range(np.size(idx_hits[0])): #don't want to pick a location we've already hit
        counts[idx_hits[0][i],idx_hits[1][i]] = 0
    Ts = transition_probability(counts)
    idxs = np.argmax(Ts) #if we wanted our rollout policy to guess the maximum probability of a hit
    n = 1; #number of trials
    a = np.array([])
    if len(args) > 0:
        a = args[0]
    else:
        i,j = np.unravel_index(Ts.argmax(), Ts.shape)
        a = np.array([i,j])
    p = Ts[a[0],a[1]] #probabiliy of success
    b._boardHitMiss[a[0],a[1]] = 1 + np.random.binomial(n, p) #hit = 2, miss = 1, not guessed = 0
    return b,r

def rollout(b,gamma,d):
    if (d <= 0) | (np.sum(b._boardHitMiss == 0) == 0):
        return 0
    b,r = state_action_sim(b)
    return r + gamma*rollout(b,gamma,d - 1)

def rewards(s):
    total_guessed = np.sum(s > 0)
    if total_guessed == 0:
        return 0
    else:
        return np.sum(s == 2)/np.sum(s > 0) #number of hits over number of guesses

def simulated_counts(b):
    counts = np.zeros(np.shape(b._boardHitMiss))
    num_samples = 10000
    for i in range(num_samples):
        available_space = (b._boardHitMiss != 1)
        for ship_length, num_ships in b.ship_sizes.items(): #will work better if ship sizes is organized in order of largests ship first
            for ship in range(num_ships):
                x1,y1,x2,y2 = try_to_place(ship_length,available_space,b)
                if (x1 != x2) | (y1 != y2):
                    counts[y1:y2+1,x1:x2+1] += 1 #upper leftmost corner is x,y = 0,0
                    available_space[y1:y2+1,x1:x2+1] = 0
    return counts

# Notes: if can not place with hits (i.e. fully surrounded by misses) need way of guessing non-hits
# need a way to check that we don't try to add a ship if it means that tot_num_ship_points < hits + current_ship_length
# handle edge case where nothing is placed and probability is all Nan or zero
# fix probability s.t. no divide by zero
def try_to_place(ship_length,available_space,b):
    hits_left = ((b._boardHitMiss == 2) & (available_space == 1))
    idx_hits_left = np.where(hits_left == 1)
    idx_available_space = np.where(available_space == 1)

    x1 = 0
    y1 = 0
    num_attempts = 0
    total_allowed_attempts = 20
    while num_attempts < total_allowed_attempts:
        if np.size(idx_hits_left) > 0 & (num_attempts < 0.5*total_allowed_attempts):
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
            if (x2 < b.board_width) & (np.sum(available_space[y1:y2+1,x1:x2+1]) == ship_length): 
                return x1,y1,x2,y2
        elif orientation == 1: #go down
            x2 = x1 
            y2 = y1 + ship_length - 1
            if (y2 < b.board_height) & (np.sum(available_space[y1:y2+1,x1:x2+1]) == ship_length): 
                return x1,y1,x2,y2
        elif orientation == 2: #go left 
            x2 = x1 - (ship_length - 1)
            y2 = y1 
            if (x2 >= 0) & (y2 < b.board_height) & (np.sum(available_space[y1:y2+1,x2:x1+1]) == ship_length): 
                return x2,y1,x1,y2
        else: #go up
            x2 = x1 
            y2 = y1 - (ship_length - 1)
            if (y2 >= 0) & (np.sum(available_space[y2:y1+1,x1:x2+1]) == ship_length): 
                return x1,y2,x2,y1
        num_attempts += 1
    return x1,y1,x1,y1

def transition_probability(counts):
    return counts/np.sum(counts)