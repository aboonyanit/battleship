import battleship
import simulate
import numpy as np


def main():
    print('Ship Locations:')
    board_width = 4
    board_height = 4
    ship_sizes = {3:1,2:1}
    b = battleship.Battleship(board_width,board_height, ship_sizes)
    b.generateRandomBoard()

    # a = [1,1] #example of an action
    # b,r = simulate.state_action_sim(b,a)

    gamma = 0.999
    d = 16
    Q = np.zeros((board_width*board_height, board_width*board_height))
    N = np.zeros((board_width*board_height, board_width*board_height))
    A = []
    for i in range(board_width):
        for j in range(board_height):
            A.append((i,j))
    r = simulate.rollout(b._boardHitMiss,gamma,d)
    s = b._boardHitMiss
    MCTS(s, board_width, board_height, ship_sizes, c=1, d=10, discount_factor=0.95, k_max=10)

def MCTS(s, board_width, board_height, ship_sizes, c, d, discount_factor, k_max):
    A = []
    for i in range(board_width):
        for j in range(board_height):
            A.append((i,j))
    Q = {}
    N = {}
    for k in range(k_max):
        # Q = np.zeros((board_width*board_height, board_width*board_height)) #why are these dimensions? 3 **, 
        # N = np.zeros((board_width*board_height, board_width*board_height))
        simulate(d, s, A, discount_factor, c, Q, N, ship_sizes)
    return Q

# Add function to map State (number) to the board
    
def simulate(d, s, A, discount_factor, c, Q, N, ship_sizes):
    if d <= 0:
        return 0
    if (s, A[0]) not in N.keys():
        for a in A:
            N[(s, a)] = 0
            Q[(s, a)] = 0
        return simulate.rollout(s, discount_factor, d)
    # Q = np.zeros((num_states, num_actions))
    # N = np.zeros((num_states, num_actions))
    a = simulate.explore(A, N, Q, c, s)
    s_prime, r = simulate.state_action_sim(s, ship_sizes, a)
    q = r + discount_factor * simulate(d - 1, s_prime, A, discount_factor, c, Q, N, ship_sizes)
    N[(s, a)] += 1
    Q[(s, a)] += (q - Q[(s, a)]) / N[(s, a)]
    return q

if __name__ == '__main__':
    main()