import heapq
import bisect

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

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id  # Player Identifier

    def play(self, board: HexBoard) -> tuple:
        raise NotImplementedError("¡Implementa este método!")
    
class Play(Player):
    
    
    
    bridge_directions = [(1,1), (1,-2), (-1,-1), (-1,2), (-2,1), (2,-1)]
    between = [((0,1),(1,0)), ((0,-1),(1,-1)), ((0,-1),(-1,0)), ((0,1),(-1,1)), ((-1,0),(-1,1)), ((1,0),(1,-1))]    # Positions between bridges
    
    neighbors = [
            (-1, 0), (1, 0),
            (0,-1), (0,1),
            (-1,1), (1,-1)
        ]
    strong_directions_player2 = [(-1,0), (-1,1), (1,0), (1,-1), (-2,1), (2,-1)]
    strong_directions_player1 = [(0,1), (0,-1), (1,1), (1,-2), (-1,-1), (-1,2), (-1,0), (-1,1), (1,0), (1,-1)]
    up_bridge = (-2,1)
    down_bridge = (2,-1) 
        
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.tt = {}
        self.player_positions = []
        self.opponente_positions = []
        self.moves_done = 0
        
    def place(self, board: HexBoard, i, j, player_id):          # Pero este no es el que se va a llamar, al principio hay que actualizar las positions
        board.place_piece(i, j, player_id)
        self.moves_done += 1
        if self.player_id == player_id:
            index = bisect.bisect(self.player_positions, (i,j), key=lambda x: x[1])     # No considera que la ia vaya de arriba a abajo
            self.player_positions.insert(index, (i,j)) 
        else:
            index = bisect.bisect(self.opponente_positions, (i,j), key=lambda x: x[0])
            self.opponente_positions.insert(index, (i,j)) 
        
    def erase_position(self, board: HexBoard, i, j):
        if self.player_id == board.board[i][j]: self.player_positions.remove((i,j))
        else: self.opponente_positions.remove((i,j))
        board.board[i][j] = 0
        self.moves_done -= 1
        
        
    def board_hash(self, board: HexBoard):
        return tuple(tuple(row) for row in board.board)
    
    def play(self, board: HexBoard) -> tuple:
        
        best = -float('inf') if self.player_id == 1 else float('inf')
        best_move = None
        #max_depth = self.moves_done // board.size + 2   # Minimax Max depth 
        max_depth = 2
        
        possible_moves = board.get_possible_moves()

        for move in possible_moves:     # For each first move I can do in this situation...
            row, col = move
            board.place_piece(row, col, self.player_id)
            self.moves_done+=1
            self.player_positions.append((row, col))
            value = self.minimax(board, max_depth - 1, -float('inf'), float('inf'), not self.player_id == 1)    # I apply minimax, so it begins at oponent's turn
            board.board[row][col]= 0
            self.moves_done-=1
            self.player_positions.remove((row,col))

            if self.player_id == 1:
                if value > best:
                    best = value
                    best_move = move
            else:
                if value < best:
                    best = value
                    best_move = move
        self.moves_done +=1
        row, col= best_move
        self.player_positions.append((row, col))
        return best_move

    def minimax(self, board: HexBoard, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        
        state = self.board_hash(board)
        if state in self.tt:
            return self.tt[state]

        # Condición de término: profundidad 0 o tablero terminal.
        if depth == 0 or board.check_connection(1) or board.check_connection(2):
            val = self.evaluate(board)
            self.tt[state] = val
            return val

        possible_moves = board.get_possible_moves()
        player_id = 1 if maximizing else 2
        if maximizing:
            best = -float('inf')
            for move in possible_moves:
                row, col = move
                board.place_piece(row, col, player_id)
                self.moves_done+=1
                self.player_positions.append((row, col))
                value = self.minimax(board, depth - 1, alpha, beta, False)
                board.board[row][col]= 0
                self.moves_done-=1
                self.player_positions.remove((row, col))
                best = max(best, value)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break  # Poda alfa-beta
            self.tt[state] = best
            return best
        else:
            best = float('inf')
            for move in possible_moves:
                row, col = move
                board.place_piece(row, col, player_id)
                self.moves_done+=1
                self.opponente_positions.append((row, col))
                value = self.minimax(board, depth - 1, alpha, beta, True)
                board.board[row][col]= 0
                self.moves_done-=1
                self.opponente_positions.remove((row, col))
                best = min(best, value)
                beta = min(beta, best)
                if alpha >= beta:
                    break  # Poda alfa-beta
            self.tt[state] = best
            return best


    def evaluate(self, board: HexBoard) -> float:
        if board.check_connection(1):
            return 10000
        if board.check_connection(2):
            return -10000
        
        # Shortest path cost
        cost_player_1 = self.shortest_path_cost(board, 1)
        cost_player_2 = self.shortest_path_cost(board, 2)
         
        center_control_1 = self.control_areas(board, 1) 
        center_control_2 = self.control_areas(board, 2)
        
        bridege_score_1 = self.bridge_score(board, 1)
        bridege_score_2 = self.bridge_score(board, 2) 
        
        cost_factor = 1 if self.moves_done < board.size - 1 else 100
            
        blocked_1 = self.blocked_positions(board, 1)
        blocked_2 = self.blocked_positions(board, 2)
    
            
        if self.player_id == 1:
            #print(f'Cost_player1: {cost_player_2 - cost_player_1}, Center_control: {center_control_1} and Bridge_score: {bridege_score_1}')
            return (cost_player_2 - cost_player_1) * cost_factor + center_control_1 * 0.1 + (bridege_score_1 - bridege_score_2) + blocked_1
        else:
            #print(f'Cost_player2: {cost_player_1 - cost_player_2}, Center_control: {center_control_2} and Bridge_score: {bridege_score_2}')
            return (cost_player_1 - cost_player_2) * cost_factor + center_control_2 * 0.1 + (bridege_score_2 - bridege_score_1) + blocked_2


    def is_bridge(self, index, board: HexBoard, i, j):
        #count = 0        
        #for d in self.bridge_directions:
        #    if(i1 + d[0] == i2 and j1 + d[1] == j2):
        bet = self.between[index]
        if board.board[i+bet[0][0]][j+bet[0][1]] == 0 and board.board[i+bet[1][0]][j+bet[1][1]] == 0:
            return True
        return False

    def bridge_score(self, board: HexBoard, player_id):
        def legal(i, j, size):
            return 0 <= i < size and 0 <= j < size
        
        score = 0
        size= board.size
        
        for pos in self.player_positions if player_id == self.player_id else self.opponente_positions:
            i,j = pos
            if board.board[i][j] == player_id:
                count = 0
                for d in self.bridge_directions:
                    bet = self.between[count]            
                    if legal(i+d[0], j+d[1], board.size) and board.board[i+d[0]][j+d[1]]:
                        if  board.board[i+bet[0][0]][j+bet[0][1]] == 0 and board.board[i+bet[1][0]][j+bet[1][1]] == 0:    
                            score += 2
                            break
                    count+=1
                    
        return score
    
    def blocked_positions(self, board: HexBoard, jugador: int) -> float:
        oponente = 2 if jugador == 1 else 1
        penalizacion = 0

        for pos in self.player_positions:
            i,j = pos
            # Contar vecinos aliados y enemigos
            vecinos_aliados = 0
            vecinos_oponentes = 0
            
            # Direcciones de vecinos en Hex (6 vecinos)
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < board.size and 0 <= nj < board.size:
                    if board.board[ni][nj] == jugador:
                        vecinos_aliados += 1
                    elif board.board[ni][nj] == oponente:
                        vecinos_oponentes += 1
            
            # Penalizar si enemigos superan a aliados por 2 o más
            if vecinos_oponentes - vecinos_aliados >= 2:
                penalizacion += 1  # Penalización por celda vulnerable

        return -penalizacion * 2  # Peso ajustable (ej: -0.5 por celda) 
    
    def shortest_path_cost(self, board: HexBoard, player_id: int) -> float:
        # Computes the shortest path for each player to connect the two sides by Dijkstra
         
        def cell_cost(cell: int, player_id: int) -> int:
            return 0 if cell == player_id else 1 if cell == 0 else 10

        size = board.size
        dist = { (i, j): float('inf') for i in range(size) for j in range(size) }
        pq = []

        # Initialization
        if player_id == 1:
            for i in range(size):
                dist[(i, 0)] = cell_cost(board.board[i][0], player_id)
                heapq.heappush(pq, (dist[(i,0)], (i, 0)))
        else:
            for j in range(size):
                dist[(0, j)] = cell_cost(board.board[0][j], player_id)
                heapq.heappush(pq, (dist[(0, j)], (0,j)))

        # Dijkstra's body
        while pq:
            current_cost, (i, j) = heapq.heappop(pq)
            if dist[(i, j)] < current_cost:
                continue
            if player_id == 1 and j == size - 1:
                return current_cost
            if player_id == 2 and i == size - 1:
                return current_cost
            for ni, nj in self.get_neighbors(i, j, size):
                new_cost = current_cost + cell_cost(board.board[ni][nj], player_id)
                if new_cost < dist[(ni, nj)]:
                    dist[(ni, nj)] = new_cost
                    heapq.heappush(pq, (new_cost, (ni, nj)))
        return float('inf')

    def control_areas(self, board: HexBoard, player_id: int) -> int:
        # Rewards strong poistions (near the center)
        
        
        def legal(i, j, size):
            return 0 <= i < size and 0 <= j < size
        
        size = board.size
        
        center = size // 2
        score = 0
        for i in range(size):
            for j in range(size):
                if board.board[i][j] == player_id:
                    score += max(0, center - abs(center - i) - abs(center - j))
        return score
    
     
    def neighbor_count(self, board: HexBoard, player_id: int) -> int:
       
        size = board.size
        score = 0

        def legal(i, j, size):
            return 0 <= i < size and 0 <= j < size

        for i in range(size):
            for j in range(size):
                if board.board[i][j] == player_id: 
                    same_color_neighbors = 0
                    for ni, nj in self.get_neighbors(i, j, size):
                        if legal(ni, nj, size) and board.board[ni][nj] == player_id:
                            same_color_neighbors += 1
                    score += same_color_neighbors * 5 

        return score


    def get_neighbors(self, i: int, j: int, board_size: int) -> list:
        # Gives the neighbors positions
        neighbors = [
            (i - 1, j), (i + 1, j),
            (i, j - 1), (i, j + 1),
            (i - 1, j + 1), (i + 1, j - 1)
        ]
        return [(ni, nj) for ni, nj in neighbors if 0 <= ni < board_size and 0 <= nj < board_size]



