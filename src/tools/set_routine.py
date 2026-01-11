import pygame
import sys
import os
import csv

WIDTH, HEIGHT = 900, 500
FPS = 60

# Paleta de colores
VERY_LIGHT_PINK = (255, 237, 237)
LIGHT_PINK = (255, 209, 221)
PINK = (238, 105, 131)
DARK_PINK = (133, 14, 53)

TITLE = "Set your workout routine"

# CSV path
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "repetitions.csv")

# Filas (label mostrado, clave interna)
rows = [
    ("Squats", "squats"),
    ("Push-ups", "pushups"),
    ("Jumps", "jumps")
]

DEFAULT_VALUE = 5

# funciones auxiliares para actualizar repetitions.csv
def ensure_csv_exists(path: str, keys, default_value: int = 5):
    """Crea carpeta/CSV si no existe, con valores por defecto."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        data = {k: default_value for k in keys}
        write_csv(path, data)


def read_csv(path: str, keys, default_value: int = 5) -> dict:
    """Lee el CSV y devuelve dict. Si falta algo, lo rellena"""
    data = {k: default_value for k in keys}
    if not os.path.exists(path):
        return data

    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Esperamos columnas: exercise,reps
            for row in reader:
                ex = (row.get("exercise") or "").strip()
                reps_raw = (row.get("reps") or "").strip()
                if ex in data:
                    try:
                        reps = int(reps_raw)
                        data[ex] = max(0, reps)
                    except ValueError:
                        # si no se puede parsear, dejamos default
                        pass
    except Exception:
        # Si el archivo está corrupto o algo falla, devolvemos defaults
        pass

    return data


def write_csv(path: str, data: dict):
    """Sobrescribe el CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["exercise", "reps"])
        writer.writeheader()
        for exercise, reps in data.items():
            writer.writerow({"exercise": exercise, "reps": int(reps)})


def save_one_change(path: str, state: dict, exercise_key: str, new_value: int):
    """Actualiza el csv"""
    state[exercise_key] = max(0, int(new_value))
    write_csv(path, state)


# Clases y métodos para pygame
class Button:
    def __init__(self, rect, label, font):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.font = font
        self.hovered = False

    def draw(self, screen):
        color = PINK if self.hovered else LIGHT_PINK
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, PINK, self.rect, width=2, border_radius=10)

        txt = self.font.render(self.label, True, DARK_PINK)
        txt_rect = txt.get_rect(center=self.rect.center)
        screen.blit(txt, txt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


def draw_panel(screen, panel_rect):
    pygame.draw.rect(screen, VERY_LIGHT_PINK, panel_rect, border_radius=18)
    pygame.draw.rect(screen, PINK, panel_rect, width=2, border_radius=18)


# Función principal
def set_routine():
    """
    Displays a Pygame window to set a workout routine using + / - buttons.

    Each exercise has its own counter. Any change made
    with the buttons is immediately saved to a CSV file.

    Notes:
    - Creates the CSV file if it does not exist.
    - Values are always >= 0.
    - Uses global constants (WIDTH, HEIGHT, FPS, TITLE, colors, etc.).
    - Exits the program when the window is closed.

    Returns:
        None
    """

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Workout Routine (Pygame)")
    clock = pygame.time.Clock()

    # Fuentes
    title_font = pygame.font.SysFont("arial", 36, bold=True)
    row_font = pygame.font.SysFont("arial", 28)
    btn_font = pygame.font.SysFont("arial", 28, bold=True)
    num_font = pygame.font.SysFont("arial", 28, bold=True)

    # Preparar CSV
    keys = [k for _, k in rows]
    ensure_csv_exists(CSV_PATH, keys, DEFAULT_VALUE)
    state = read_csv(CSV_PATH, keys, DEFAULT_VALUE)

    # Panel central
    panel_w, panel_h = 760, 380
    panel_rect = pygame.Rect((WIDTH - panel_w) // 2, (HEIGHT - panel_h) // 2, panel_w, panel_h)

    # Layout
    title_y = panel_rect.y + 30
    left_x = panel_rect.x + 70
    controls_x = panel_rect.right - 240
    row_start_y = panel_rect.y + 120
    row_gap = 80

    btn_size = (48, 42)
    minus_offset_x = 0
    plus_offset_x = 140

    # Crear botones por fila
    ui = {}
    for i, (label, key) in enumerate(rows):
        y = row_start_y + i * row_gap
        minus_rect = (controls_x + minus_offset_x, y - btn_size[1] // 2, btn_size[0], btn_size[1])
        plus_rect = (controls_x + plus_offset_x,  y - btn_size[1] // 2, btn_size[0], btn_size[1])

        ui[key] = {
            "label": label,
            "minus": Button(minus_rect, "-", btn_font),
            "plus": Button(plus_rect, "+", btn_font),
            "y": y
        }

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Interacción botones y guardado en CSV
            for key in ui:
                if ui[key]["minus"].handle_event(event):
                    new_value = max(0, state[key] - 1)
                    save_one_change(CSV_PATH, state, key, new_value)

                if ui[key]["plus"].handle_event(event):
                    new_value = state[key] + 1
                    save_one_change(CSV_PATH, state, key, new_value)

        # Hover en botones
        mouse_pos = pygame.mouse.get_pos()
        for key in ui:
            ui[key]["minus"].hovered = ui[key]["minus"].rect.collidepoint(mouse_pos)
            ui[key]["plus"].hovered = ui[key]["plus"].rect.collidepoint(mouse_pos)

        # Draw
        screen.fill(LIGHT_PINK)
        draw_panel(screen, panel_rect)

        # Título centrado
        title_surf = title_font.render(TITLE, True, DARK_PINK)
        title_rect = title_surf.get_rect(midtop=(panel_rect.centerx, title_y))
        screen.blit(title_surf, title_rect)


        # Filas
        for key, parts in ui.items():
            y = parts["y"]
            label = parts["label"]

            # Texto izquierda
            label_surf = row_font.render(label, True, DARK_PINK)
            label_rect = label_surf.get_rect(midleft=(left_x, y))
            screen.blit(label_surf, label_rect)

            # Botones
            parts["minus"].draw(screen)
            parts["plus"].draw(screen)

            # Número entre botones
            value = state[key]
            num_surf = num_font.render(str(value), True, DARK_PINK)
            num_center_x = parts["minus"].rect.centerx + (parts["plus"].rect.centerx - parts["minus"].rect.centerx) // 2
            num_rect = num_surf.get_rect(center=(num_center_x, y))
            screen.blit(num_surf, num_rect)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    set_routine()
