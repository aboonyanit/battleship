import battleship
import simulate


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
    r = simulate.rollout(b,gamma,d)
    


if __name__ == '__main__':
    main()