import tkinter as tk
import math
from hex_table import HexBoard
from player import Play

class HexGameGUI:
    def __init__(self, board_size):
        self.board_size = board_size
        self.hex_radius = 40      
        self.margin = 80           
        self.board = HexBoard(board_size)
        self.ai_player = Play(1)
        self.human_player_id = 2
        self.current_player = None 
        self.game_over = False

        self.window = tk.Tk()
        self.window.title("Juego Hex: Humano vs IA")
        # Cálculo de ancho y alto para el diseño pointy-topped
        self.canvas_width = int(self.margin * 2 + self.hex_radius * math.sqrt(3) * (board_size + 0.5))
        self.canvas_height = int(self.margin * 2 + self.hex_radius * 1.5 * (board_size - 1) + self.hex_radius * 2)
        self.canvas = tk.Canvas(self.window, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()
        
        # Vincula el clic en el Canvas
        self.canvas.bind('<Button-1>', self.on_canvas_click)

        self.status_label = tk.Label(self.window, text="", font=("Arial", 16))
        self.status_label.pack(pady=10)

        self.restart_button = tk.Button(self.window, text="Reiniciar Juego", font=("Arial", 12), command=self.restart_game)
        self.restart_button.pack(pady=5)

        self.cell_polygons = {}  # Diccionario para almacenar los hexágonos
        self.choose_starter()   # Pantalla inicial para elegir quién comienza

    def choose_starter(self):
        """Muestra una ventana para elegir quién comienza."""
        starter_window = tk.Toplevel(self.window)
        starter_window.title("Elegir Quién Comienza")
        starter_window.geometry("300x200")
        tk.Label(starter_window, text="¿Quién comienza?", font=("Arial", 14)).pack(pady=20)
        tk.Button(starter_window, text="Humano", font=("Arial", 12),
                  command=lambda: self.start_game(starter_window, "human")).pack(pady=10)
        tk.Button(starter_window, text="IA", font=("Arial", 12),
                  command=lambda: self.start_game(starter_window, "ai")).pack(pady=10)

    def start_game(self, starter_window, first_player):
        """Inicia el juego según la elección."""
        starter_window.destroy()
        if first_player == "human":
            self.current_player = self.human_player_id
            self.status_label.config(text="Turno: Humano (Azul)")
        else:
            self.current_player = self.ai_player.player_id
            self.status_label.config(text="Turno: IA (Rojo)")
            self.window.after(500, self.ai_move)
        self.draw_board()

    def draw_board(self):
        """Dibuja el tablero: bordes decorativos y hexágonos en retícula."""
        self.canvas.delete("all")
        self.cell_polygons = {}

        # (Opcional) Dibujar bordes decorativos.
        self.canvas.create_line(
            self.margin/2, self.margin,
            self.margin/2, self.canvas_height - self.margin,
            fill="red", width=10
        )
        self.canvas.create_line(
            self.canvas_width - self.margin/2, self.margin,
            self.canvas_width - self.margin/2, self.canvas_height - self.margin,
            fill="red", width=10
        )
        self.canvas.create_line(
            self.margin, self.margin/2,
            self.canvas_width - self.margin, self.margin/2,
            fill="blue", width=10
        )
        self.canvas.create_line(
            self.margin, self.canvas_height - self.margin/2,
            self.canvas_width - self.margin, self.canvas_height - self.margin/2,
            fill="blue", width=10
        )

        # Dibujar los hexágonos usando el modelo pointy-topped:
        for row in range(self.board_size):
            for col in range(self.board_size):
                cx, cy = self.get_hex_center(row, col)
                points = self.hexagon_points(cx, cy, self.hex_radius)
                fill_color = "white"
                if self.board.board[row][col] == 1:
                    fill_color = "red"
                elif self.board.board[row][col] == 2:
                    fill_color = "blue"
                tag = f"cell_{row}_{col}"
                poly_id = self.canvas.create_polygon(points, fill=fill_color, outline="black", width=2, tags=tag)
                self.cell_polygons[(row, col)] = poly_id

    def get_hex_center(self, row, col):
        """
        Calcula el centro de un hexágono en la posición (row, col).
        Cada fila está desplazada horizontalmente una posición más a la derecha
        respecto a la fila anterior.
        """
        cx = self.margin + self.hex_radius * math.sqrt(3) * col + (self.hex_radius * math.sqrt(3) / 2) * row
        cy = self.margin + self.hex_radius * 1.5 * row + self.hex_radius
        return cx, cy



    def hexagon_points(self, cx, cy, r):
        """
        Devuelve la lista de coordenadas para dibujar un hexágono con orientación pointy-topped.
        Se utiliza la fórmula:
          angulo = 60 * i - 30,  i = 0...5.
        """
        points = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.radians(angle_deg)
            x = cx + r * math.cos(angle_rad)
            y = cy + r * math.sin(angle_rad)
            points.extend([x, y])
        return points

    def on_canvas_click(self, event):
        """Detecta la posición clicada y traduce las coordenadas a (row, col) en la matriz lógica."""
        if self.game_over:
            return

        clicked_x, clicked_y = event.x, event.y

        # Convertir las coordenadas (x, y) al hexágono más cercano en la matriz lógica
        for row in range(self.board_size):
            for col in range(self.board_size):
                cx, cy = self.get_hex_center(row, col)  # Centro del hexágono en la posición (row, col)
                if self.is_point_in_hexagon(clicked_x, clicked_y, cx, cy):
                    # Si el clic cae dentro del hexágono, manejar el movimiento
                    self.handle_human_move(row, col)
                    return
                
    def is_point_in_hexagon(self, x, y, cx, cy):
        """
        Verifica si el punto (x, y) está dentro de un hexágono centrado en (cx, cy) con orientación pointy-topped.
        """
        dx = abs(x - cx)
        dy = abs(y - cy)
        if dx > self.hex_radius * math.sqrt(3) / 2 or dy > self.hex_radius:
            return False  # Está fuera de los límites del hexágono de forma evidente

        # Fórmula para verificar si está dentro del hexágono
        return dx * math.sqrt(3) + dy <= self.hex_radius * math.sqrt(3)


    def handle_human_move(self, row, col):
        """Realiza el movimiento del humano, actualiza el tablero y verifica la victoria."""
        if self.board.board[row][col] != 0:
            return
        self.board.place_piece(row, col, self.human_player_id)
        self.draw_board()
        if self.board.check_connection(self.human_player_id):
            self.status_label.config(text="¡Ganaste (Humano - Azul)!")
            self.game_over = True
            return
        if not self.board.get_possible_moves():
            self.status_label.config(text="¡Empate!")
            self.game_over = True
            return
        self.status_label.config(text="Turno: IA...")
        self.window.after(500, self.ai_move)

    def ai_move(self):
        if self.game_over:
            return
        move = self.ai_player.play(self.board)
        if move:
            row, col = move
            self.board.place_piece(row, col, self.ai_player.player_id)
            #print(f'Bridge score: {self.ai_player.bridge_score(self.board, 1) - self.ai_player.bridge_score(self.board, 2)}')
            self.draw_board()
            if self.board.check_connection(self.ai_player.player_id):
                self.status_label.config(text="Gana la IA (Rojo)!")
                self.game_over = True
                return
        if not self.board.get_possible_moves():
            self.status_label.config(text="¡Empate!")
            self.game_over = True
            return
        self.status_label.config(text="Turno: Humano (Azul)")

    def restart_game(self):
        self.board = HexBoard(self.board_size)
        self.game_over = False
        self.choose_starter()

if __name__ == "__main__":
    game = HexGameGUI(board_size=11)
    game.window.mainloop()
