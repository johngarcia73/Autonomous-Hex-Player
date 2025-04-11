import heapq
#import time
from functools import lru_cache
from hex_table import HexBoard

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
        self.player_positions = set()  
        self.opponente_positions = set() 
        self.moves_done = 0
        self.precomputed_neighbors = {} 
        self.zobrist_table = {}  
        self.current_hash = 0  
        self.depth_cache = {} 
        self.move_history = [] 
        #self.start_time = 0
        #self.time_limit = 5
    
    def init_zobrist(self, size):
        import random
        self.zobrist_table = {(i,j,p): random.getrandbits(64) 
                             for i in range(size) 
                             for j in range(size) 
                             for p in [0,1,2]}
    
    def board_hash(self, board: HexBoard):
        if not self.zobrist_table:
            self.init_zobrist(board.size)
        h = 0
        for i in range(board.size):
            for j in range(board.size):
                h ^= self.zobrist_table[(i,j,board.board[i][j])]
        return h    
    
    @lru_cache(maxsize=None)
    def get_neighbors(self, i: int, j: int, board_size: int):
        return [(i+di, j+dj) for di, dj in self.neighbors 
                if 0 <= i+di < board_size and 0 <= j+dj < board_size]
    
    def place(self, board: HexBoard, i, j, player_id):
        board.place_piece(i, j, player_id)
        self.moves_done += 1
        target_set = self.player_positions if player_id == self.player_id else self.opponente_positions
        target_set.add((i,j))
    
    def erase_position(self, board: HexBoard, i, j):
        target_set = self.player_positions if board.board[i][j] == self.player_id else self.opponente_positions
        target_set.discard((i,j))
        board.board[i][j] = 0
        self.moves_done -= 1
    
    def play(self, board: HexBoard) -> tuple:
        #self.time_limit = time_limit
        #self.start_time = time.time()
        
        if not hasattr(self, 'tt'): self.tt = {}
        
        if not self.zobrist_table:
            self.init_zobrist(board.size)
            self.current_hash = self.board_hash(board)
        
        
        best_value = -float('inf') if self.player_id == 1 else float('inf')
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        max_depth = self.calculate_depth(board.size, board)

        # Obtaining ordered possible moves
        possible_moves = self.get_ordered_moves(board, True)
        
        
        for move in possible_moves:
            
            #if self.time_out(): return best_move if best_move else move
        
            i, j = move
            
            # For making the first move before minimax
            self.simulate_move(board, i, j, self.player_id)
            
            current_value = self.minimax(
                board, 
                depth=max_depth -1,
                alpha=alpha,
                beta=beta,
                maximizing=(self.player_id==2))
            
            if self.is_better_value(current_value, best_value):
                best_value = current_value
                best_move = move
                
                if self.player_id == 1:
                    alpha = max(alpha, best_value)
                else:
                    beta = min(beta, best_value)
                    
            self.undo_move(board, i, j)
            #if self.pruning_condition(alpha, beta):
            #    break
            
        # Statistics register
        self.update_game_state(best_move)
        return best_move
        

    def simulate_move(self, board: HexBoard, i: int, j: int, player: int):
        self.current_hash ^= self.zobrist_table[(i, j, board.board[i][j])]
        board.board[i][j] = player
        self.current_hash ^= self.zobrist_table[(i, j, player)]
        
        target_set = self.player_positions if player == self.player_id else self.opponente_positions
        target_set.add((i, j))
        self.moves_done += 1

    def undo_move(self, board: HexBoard, i: int, j: int): 
        player = board.board[i][j]
        
        self.current_hash ^= self.zobrist_table[(i, j, player)]
        board.board[i][j] = 0
        self.current_hash ^= self.zobrist_table[(i, j, 0)]
        
        # Actualización de posiciones
        target_set = self.player_positions if player == self.player_id else self.opponente_positions
        target_set.discard((i, j))
        self.moves_done -= 1

    def is_better_value(self, new_value: float, current_best: float) -> bool:
        return (new_value > current_best) if self.player_id == 1 else (new_value < current_best)

    def pruning_condition(self, alpha: float, beta: float) -> bool:
        return (alpha >= beta) if self.player_id == 1 else (beta <= alpha)

    def update_game_state(self, move: tuple):
        # Upgrades statistics
        if not hasattr(self, 'move_history'):
            self.move_history = []
        self.move_history.append(move)
        
        # Cleans the tt
        if hasattr(self, 'tt'):
            cleaned_tt = {}
            for k, v in self.tt.items():
                # Correct formatting
                if isinstance(v, dict) and 'depth' in v and 'value' in v:
                    if v['depth'] >= 2:
                        cleaned_tt[k] = v
            self.tt = cleaned_tt
        
    def minimax(self, board: HexBoard, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        
        
        state_hash = self.board_hash(board)
        
        # Verifies the tt
        if hasattr(self, 'tt') and state_hash in self.tt:
            entry = self.tt[state_hash]
            # Checks correct format
            if isinstance(entry, dict) and 'depth' in entry and 'value' in entry:
                if entry['depth'] >= depth:
                    return entry['value']
        
        # Win condition
        if board.check_connection(1):
            return 10000
        if board.check_connection(2):
            return -10000
        if depth == 0:
            val = self.evaluate(board)
            if not hasattr(self, 'tt'):
                self.tt = {}
            self.tt[state_hash] = {'value': val, 'depth': depth}
            return val

        possible_moves = self.get_ordered_moves(board, maximizing)
        best_value = -float('inf') if maximizing else float('inf')
        player = 1 if maximizing else 2

        for move in possible_moves:
            i, j = move
            self.place(board, i, j, player)
            value = self.minimax(board, depth-1, alpha, beta, not maximizing)
            self.erase_position(board, i, j)

            if maximizing:
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
            else:
                best_value = min(best_value, value)
                beta = min(beta, best_value)

            if alpha >= beta:
                break
            
        # Saves the result
        if not hasattr(self, 'tt'):
            self.tt = {}
        self.tt[state_hash] = {'value': best_value, 'depth': depth}
        
        return best_value
    
    #def time_out(self) -> bool:
        #return (time.time() - self.start_time) >= (self.time_limit - 0.05)  # 50ms to end stuff
    
    def evaluate(self, board: HexBoard) -> float:
        # Win condition
        if board.check_connection(1):
            return float('inf')
        if board.check_connection(2):
            return float('-inf')
        
        # Connection costs 
        cost_player_1 = self.shortest_path_cost(board, 1)
        cost_player_2 = self.shortest_path_cost(board, 2)
        
        # Central control
        center_control_1 = self.control_areas(board, 1)
        center_control_2 = self.control_areas(board, 2)
        
        # Bridges score
        bridge_score_1 = self.bridge_score(board, 1)
        bridge_score_2 = self.bridge_score(board, 2)
        
        # Game phase
        game_phase = self.calculate_game_phase(board)
        
        blocked_player1 =self.blocked_positions(board, 1)
        blocked_player2 =self.blocked_positions(board, 2)
        
        #pressure = self.pressure(board)-----
        
        # Dynamic weights by game phase  there is no midgame xd
        if game_phase == 'early':
            bridge_weight = 0.7
            path_weight = 0.3
        else:
            bridge_weight = 0.1
            path_weight = 100
        
        
        if self.player_id == 1:
            score = (
                (cost_player_2 - cost_player_1) * path_weight * 2 + 
                (center_control_1 - center_control_2) * 0.1 +
                (bridge_score_1 - bridge_score_2) * bridge_weight + blocked_player1
            )
        else:
            score = (
                (cost_player_1 - cost_player_2) * path_weight * 2 +
                (center_control_2 - center_control_1) * 0.1 +
                (bridge_score_2 - bridge_score_1) * bridge_weight +blocked_player2
            )
        
        
        return score

    def pressure(self, board: HexBoard) -> float:
        
        size = board.size
        pressure = 0
        
        row_pressure = [0.0] * size
        col_pressure = [0.0] * size
        
        for i in range(size):
            for j in range(size):
                cell = board.board[i][j]
                if cell != 0:
                    weight = 1.2 if cell == self.player_id else 1.0
                    row_pressure[i] += weight
                    col_pressure[j] += weight
                    
                    for di, dj in self.neighbors:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size:
                            row_pressure[ni] += weight * 0.3
                        if 0 <= nj < size:
                            col_pressure[nj] += weight * 0.3
     
        max_row_pressure = max(row_pressure) if row_pressure else 0
        max_col_pressure = max(col_pressure) if col_pressure else 0
        
        critical_rows = [i for i, val in enumerate(row_pressure) 
                        if val >= max_row_pressure * 0.85]
        critical_cols = [j for j, val in enumerate(col_pressure) 
                        if val >= max_col_pressure * 0.85]
        
        for i in range(size):
            for j in range(size):
                if board.board[i][j] == 0:  
                    row_proximity = min(abs(i - r) for r in critical_rows) if critical_rows else size
                    col_proximity = min(abs(j - c) for c in critical_cols) if critical_cols else size
                    
                    pressure += 1.5 / (1 + row_proximity)
                    pressure += 1.5 / (1 + col_proximity)
                    break;
        
        return pressure



    def get_critical_zones(self, pressure: list) -> list:
        if not pressure:
            return []
        
        max_pressure = max(pressure)
        threshold = max_pressure * 0.8
        return [i for i, val in enumerate(pressure) if val >= threshold]
    
    def calculate_game_phase(self, board: HexBoard) -> str:
        filled_cells = sum(1 for row in board.board for cell in row if cell != 0)
        total_cells = board.size * board.size
        
        if filled_cells*filled_cells < total_cells + 2:
            return 'early'
        else:
            return 'late'

    def calculate_depth(self, size: int, board: HexBoard) -> int:
        
        base_depths = {
            5: 10,
            7: 5,
            9: 4,
            11: 3
        }
        
        base_depth = base_depths.get(size, 3)
        
        opening_moves = size          
        lategame_moves = size * 2       
        
        if self.moves_done < opening_moves:
            phase_adjustment = -1
        elif self.moves_done < lategame_moves:
            phase_adjustment = 1
        else:
            phase_adjustment = 2 if len(self.player_positions) > size else 0
        
        complexity = 0
        if len(self.opponente_positions) > size // 2:
            complexity = self.positional_complexity(board)
        
        
        depth = base_depth + phase_adjustment + complexity
        
        
        return max(2, min(depth, 6))
    
    def positional_complexity(self, board: HexBoard) -> int:
        
        bridge_count = sum(1 for _ in self.find_potential_bridges(board))
        
        player_density = len(self.player_positions) / (board.size ** 2)
        opponent_density = len(self.opponente_positions) / (board.size ** 2)
        
        if bridge_count > 3 or abs(player_density - opponent_density) > 0.2:
            return 2
        elif bridge_count > 1:
            return 1
        return 0
    
    def check_single_bridge(self, board: HexBoard, i: int, j: int) -> int:
        #To detect one bridge
        for idx, (di, dj) in enumerate(self.bridge_directions):
            ni, nj = i + di, j + dj
            if 0 <= ni < board.size and 0 <= nj < board.size:
                if board.board[ni][nj] == board.board[i][j]:
                    bet = self.between[idx]
                    if board.board[i+bet[0][0]][j+bet[0][1]] == 0 and \
                       board.board[i+bet[1][0]][j+bet[1][1]] == 0:
                        return 2
        return 0
    
    def get_ordered_moves(self, board: HexBoard, maximizing: bool) -> list:
        # Order by strategic value
        size = board.size
        center = size // 2
        moves = []
        
        for move in board.get_possible_moves():
            i, j = move
            priority = 0
            
            # 1- Inmetieate connections
            if self.is_winning_move(board, move, 1 if maximizing else 2):
                return [move]
            
            # 2- Nearby our cells
            near_ally = any(abs(x-i) <= 2 and abs(y-j) <= 2 
                         for x, y in (self.player_positions if maximizing else self.opponente_positions))
            
            # 3- Central positions
            distance_to_center = abs(i - center) + abs(j - center)
            
            priority = (near_ally * 1000) + (50 - distance_to_center)
            moves.append((priority, move))
        
        return [m[1] for m in sorted(moves, key=lambda x: -x[0])]

    def is_winning_move(self, board: HexBoard, move: tuple, player: int) -> bool:
        i, j = move
        board.place_piece(i, j, player)
        result = board.check_connection(player)
        board.board[i][j] = 0
        return result
    
    
    def is_bridge(self, index, board: HexBoard, i, j):
        bet = self.between[index]
        if board.board[i+bet[0][0]][j+bet[0][1]] == 0 and board.board[i+bet[1][0]][j+bet[1][1]] == 0:
            return True
        return False

    def bridge_score(self, board: HexBoard, player_id):
        score = 0
        size = board.size
        positions = self.player_positions if player_id == self.player_id else self.opponente_positions
        
        for i, j in positions:
            for idx, (di, dj) in enumerate(self.bridge_directions):
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size and board.board[ni][nj] == player_id:
                    bet = self.between[idx]
                    if board.board[i+bet[0][0]][j+bet[0][1]] == 0 and \
                       board.board[i+bet[1][0]][j+bet[1][1]] == 0:
                        score += 2
                        break 
        return score
    
    def blocked_positions(self, board: HexBoard, jugador: int) -> float:
        oponente = 2 if jugador == 1 else 1
        penalizacion = 0

        for pos in self.player_positions:
            i,j = pos
            vecinos_aliados = 0
            vecinos_oponentes = 0
            
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < board.size and 0 <= nj < board.size:
                    if board.board[ni][nj] == jugador:
                        vecinos_aliados += 1
                    elif board.board[ni][nj] == oponente:
                        vecinos_oponentes += 1
            
            # Punishes if enemies overwhelm allies by twoor more
            if vecinos_oponentes - vecinos_aliados >= 2:
                penalizacion += 1

        return -penalizacion * 2 
    
    def shortest_path_cost(self, board: HexBoard, player_id: int) -> float:
        
        # Dijkstra
        size = board.size
        dist = [[float('inf')] * size for _ in range(size)]
        heap = []
        
       
        def cell_cost(i, j):
            cell = board.board[i][j]
            return 0 if cell == player_id else 1 if cell == 0 else 10
        
        if player_id == 1:
            for i in range(size):
                current_cost = cell_cost(i, 0)
                dist[i][0] = current_cost
                heapq.heappush(heap, (current_cost, i, 0))
        else:
            for j in range(size):
                current_cost = cell_cost(0, j)
                dist[0][j] = current_cost
                heapq.heappush(heap, (current_cost, 0, j))
        
        # precalculated directions, the neighbors
        dirs = self.neighbors
        
        while heap:
            current_cost, i, j = heapq.heappop(heap)
            
            # Early exit if finds objective
            if (player_id == 1 and j == size - 1) or (player_id == 2 and i == size - 1):
                return current_cost
            
            if current_cost > dist[i][j]:
                continue
            
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    new_cost = current_cost + cell_cost(ni, nj)
                    
                    if new_cost < dist[ni][nj]:
                        dist[ni][nj] = new_cost
                        heapq.heappush(heap, (new_cost, ni, nj))
        
        return float('inf')

    def control_areas(self, board: HexBoard, player_id: int) -> int:
        # Rewards strong beggining poistions (near the center)
        
        
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
    
     
    def count_neighbors(self, board: HexBoard, i: int, j: int, player: int) -> tuple:
        allies = 0
        enemies = 0
        for ni, nj in self.get_neighbors(i, j, board.size):
            if board.board[ni][nj] == player:
                allies += 1
            elif board.board[ni][nj] != 0:
                enemies += 1
        return allies, enemies




