import battleship
import simulate
import numpy as np

# board_width: int = 8,
# board_height: int = 8,
# ship_sizes = {5: 1, 4: 1, 3: 2, 2: 1} #key: ship size, value: number of ships
def main():
    # board_width = 6
    # board_height = 6
    # ship_sizes = {3:2, 2:2}
    board_width: int = 8
    board_height: int = 8
    ship_sizes = {5: 1, 4: 1, 3: 2, 2: 1} #key: ship size, value: number of ships
    b = battleship.Battleship(board_width,board_height, ship_sizes)
    b.generateRandomBoard()

    #run this to play game
    s = b._boardHitMiss
    A = []
    for i in range(board_width):
        for j in range(board_height):
            A.append((i,j))
    print(s)
    counter = 0
    tot_num_hits = np.sum([ship_length*num_ships for ship_length, num_ships in ship_sizes.items()])
    while np.sum(b._boardHitMiss > 0) < board_width*board_height  and (np.sum(s >= 2) < tot_num_hits):
        s_new=s.copy()
        a = MCTS(s_new, A, board_width, board_height, ship_sizes, c=2, d=3, discount_factor=0.5, k_max=board_width*board_height)
        print("action: ", a)
        if b._boardShipLocations[a[0],a[1]] > 0:
            ship_number = b._boardShipLocations[a[0],a[1]] 
            idxs_ship_number = np.where(b._boardShipLocations == ship_number)
            ship_length = len(idxs_ship_number[0])
            if np.sum(b._boardHitMiss[idxs_ship_number[0],idxs_ship_number[1]] == 2) == ship_length - 1:
                b._boardHitMiss[idxs_ship_number[0],idxs_ship_number[1]] = 3
            else:
                b._boardHitMiss[a[0],a[1]] = 2
        else:
            b._boardHitMiss[a[0],a[1]] = 1
        s = b._boardHitMiss
        counter = counter + 1
        print(s)
    print("Number of moves to win = ")
    print(counter)

    #run this to play with random strategy
    # A = []
    # for i in range(board_width):
    #     for j in range(board_height):
    #         A.append((i,j))
    # tot_num_hits = np.sum([ship_length*num_ships for ship_length, num_ships in b.ship_sizes.items()])
    # num_sims = 100
    # num_moves_to_win = 0
    # for j in range(num_sims):
    #     b = battleship.Battleship(board_width,board_height, ship_sizes)
    #     b.generateRandomBoard()
    #     counter = 0
    #     A_idxs = [i for i in range(int(np.size(A)/2))]
    #     while np.sum(b._boardHitMiss > 1) != tot_num_hits and np.size(A_idxs) > 0:
    #         idx = np.random.choice(A_idxs,replace=False)
    #         A_idxs.remove(idx)
    #         a = A[idx]
    #         if b._boardShipLocations[a[0],a[1]] > 0:
    #             b._boardHitMiss[a[0],a[1]] = 2
    #         else:
    #             b._boardHitMiss[a[0],a[1]] = 1
    #         counter += 1
    #         # print("Guess #" + str(counter))
    #         # print(b._boardHitMiss)
    #     # b.refreshHitMiss()
    #     num_moves_to_win += counter
    #     # print("Number of moves to win (random policy) = ")
    #     # print(counter)
    # num_moves_to_win /= num_sims
    # print("Average number of moves to win with random policy = " + str(num_moves_to_win))

def MCTS(s, A,  board_width, board_height, ship_sizes, c, d, discount_factor, k_max):

    Q = {}
    N = {}
    for k in range(k_max):
        s_copy = s.copy()
        sim(d, s_copy, A, discount_factor, c, Q, N, ship_sizes)
    max_Q_val = -float('inf')
    max_A = 0
    for i in range(len(A)):
        a = A[i]
        s_tup = tuple(map(tuple, s))
        s_action_pair = s_tup + tuple(a)
        if s_action_pair in Q.keys():
            if Q[s_action_pair] > max_Q_val:
                max_Q_val = Q[s_action_pair]
                max_A = i
    final = A[max_A]
    del A[max_A]
    return final
    # return np.argmax(Q(s,a) for a in A) #<-- idk if this will work??

# Add function to map State (number) to the board
    
def sim(d, s, A, discount_factor, c, Q, N, ship_sizes):
    if d <= 0:
        return 0
    s_tup = tuple(map(tuple, s))
    s_action_pair = s_tup + tuple(A[0])
    if s_action_pair not in N.keys():
        for a in A:
            s_action_pair = s_tup + tuple(a)
            N[s_action_pair] = 0
            Q[s_action_pair] = 0
        s_copy = s.copy()
        return simulate.rollout(s_copy, ship_sizes, discount_factor, d)
    a = simulate.explore(A, N, Q, c, s)
    s_copy = s.copy()
    s_prime, r = simulate.state_action_sim_rand3(s_copy, ship_sizes, a)
    q = r + discount_factor * sim(d - 1, s_prime, A, discount_factor, c, Q, N, ship_sizes)
    s_action_pair = s_tup + a
    N[s_action_pair] += 1
    Q[s_action_pair] += (q - Q[s_action_pair]) / N[s_action_pair]
    return q

if __name__ == '__main__':
    main()