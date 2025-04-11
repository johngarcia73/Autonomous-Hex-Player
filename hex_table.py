class HexBoard:
    def __init__(self, size: int):
        self.size = size  # Size N of the board (NxN)
        self.board = [[0] * size for _ in range(size)]  # Matrix NxN (0=empty, 1=Player1, 2=Player2)
        self.player_positions = {1: set(), 2: set()}  # Set Pieces Per Player
        
    def clone(self):
        new_board = HexBoard(self.size)
        new_board.board = [row[:] for row in self.board]
        return new_board

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        #Place a token if the square is empty
        if (self.board[row][col] == 0):
            self.board[row][col] = player_id
            return True
        return False

    def get_possible_moves(self) -> list:
        # Returns all empty boxes as tuples (row, column)
        emptyCells = []
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if(self.board[i][j] == 0):
                    emptyCells.append((i,j))
        return emptyCells

    def check_connection(self, player_id: int) -> bool:
        def get_neighbors(i, j):
            return [
                (i-1, j),
                (i-1, j + 1),   
                (i, j-1),       
                (i, j+1),       
                (i + 1, j - 1),
                (i + 1, j)     
            ]

        def legal(i, j):
            return 0 <= i < self.size and 0 <= j < self.size

        def dfs(i, j, visited):
            if (player_id == 1 and j == self.size - 1) or (player_id == 2 and i == self.size - 1):
                return True

            visited.add((i, j))
            for ni, nj in get_neighbors(i, j):
                if legal(ni, nj) and (ni, nj) not in visited and self.board[ni][nj] == player_id:
                    if dfs(ni, nj, visited):
                        return True
            return False

        visited = set()
        if player_id == 1:
            for i in range(self.size):
                if self.board[i][0] == player_id and dfs(i, 0, visited):
                    return True
        elif player_id == 2:
            for j in range(self.size):
                if self.board[0][j] == player_id and dfs(0, j, visited):
                    return True

        return False

 