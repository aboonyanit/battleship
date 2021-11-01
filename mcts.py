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
    r = simulate.rollout(b,gamma,d)
    


if __name__ == '__main__':
    main()