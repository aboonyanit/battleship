import random
import numpy as np

class Battleship():
  def __init__(
    self,
    board_width: int = 8,
    board_height: int = 8,
    ship_sizes = {5: 1, 4: 1, 3: 2, 2: 1} #key: ship size, value: number of ships
  ):
    self.board_width = board_width
    self.board_height = board_height
    self._boardShipLocations = np.zeros((self.board_height, self.board_width), dtype=np.int8) #ship locations
    self._boardHitMiss = np.zeros((self.board_height, self.board_width), dtype=np.int8) #ship locations
    self.ship_sizes = ship_sizes

  def generateRandomBoard(self):
    # 0's if no ship and 1 if ship
    for key, value in self.ship_sizes.items():
      #generate random orientation (across or downards)
      for j in range(value):
        isPiecePlaced = False
        while not isPiecePlaced:
          orientation = random.randrange(2)
          x_coord = random.randrange(self.board_width)
          y_coord = random.randrange(self.board_height)
          if orientation == 0:
            # go right across
            isBoardValid = True
            for i in range(key):
              if y_coord + i >= self.board_height:
                isBoardValid = False
                break
              if self._boardShipLocations[x_coord][i + y_coord] == 1:
                isBoardValid = False
                break
            if isBoardValid:
              for i in range(key):
                self._boardShipLocations[x_coord][i + y_coord] = 1
              isPiecePlaced = True
          else:
            # go down
            isBoardValid = True
            for i in range(key):
              if x_coord + i >= self.board_width:
                isBoardValid = False
                break
              if self._boardShipLocations[x_coord + i][y_coord] == 1:
                isBoardValid = False
                break
            if isBoardValid:
              for i in range(key):
                self._boardShipLocations[x_coord + i][y_coord] = 1
              isPiecePlaced = True
    print("Ship Locations")
    print(self._boardShipLocations)
    return self._boardShipLocations

  def refreshHitMiss(self):
    self._boardHitMiss = np.zeros((self.board_height,self.board_width), dtype=np.int8)
    return self._boardHitMiss

Battleship().generateRandomBoard()