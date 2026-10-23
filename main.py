"""
LASER LOGIC — webcam-controlled neon pipe puzzle.
Run: python main.py
"""
import json
import math
import random
import shutil
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

try:
    import pygame
except ImportError:
    pygame = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None

WIDTH, HEIGHT = 1100, 760
FPS = 60
DWELL_SECONDS = 0.5
BG = (8, 12, 29)
PANEL = (15, 23, 48)
PANEL2 = (20, 31, 62)
CYAN = (57, 232, 255)
BLUE = (65, 127, 255)
PURPLE = (157, 91, 255)
MAGENTA = (255, 70, 180)
WHITE = (231, 242, 255)
MUTED = (128, 151, 184)
GREEN = (89, 255, 177)
RED = (255, 90, 112)
PIPE_DIM = (72, 96, 140)   # pipe with no laser in it
PIPE_EDGE = (6, 10, 24)    # dark outline around every pipe

# Direction vectors are row/column; bit flags make reciprocal checks cheap.
N, E, S, W = 1, 2, 4, 8
DIRS = [(N, -1, 0, S), (E, 0, 1, W), (S, 1, 0, N), (W, 0, -1, E)]
STRAIGHT = [N | S, E | W]
CORNER = [N | E, E | S, S | W, W | N]
ALL_PIPES = STRAIGHT + CORNER
DATA_FILE = Path(__file__).with_name("high_score.json")


def rotate_mask(mask, turns=1):
    """Rotate N/E/S/W bit mask clockwise by quarter-turns."""
    turns %= 4
    for _ in range(turns):
        mask = ((mask << 1) & 15) | ((mask >> 3) & 1)
    return mask


def rotate_tile(tile):
    """Pure helper used by the game and verification checks."""
    return rotate_mask(tile, 1)


def connected(a, b, direction):
    opposite = {N: S, E: W, S: N, W: E}[direction]
    return bool(a & direction) and bool(b & opposite)


class Puzzle:
    """Solvable board built from a self-avoiding source-to-exit path."""
    def __init__(self, size=5, seed=None):
        self.rng = random.Random(seed)
        self.size = size
        self.source = (self.rng.randrange(size), 0)
        self.exit = (self.rng.randrange(size), size - 1)
        self.tiles = [[0 for _ in range(size)] for _ in range(size)]
        self.solution = {}
        self.scramble = {}
        self.generate()

    def base_path(self, start, goal):
        """Simple monotone route: across, down/up the exit row, across."""
        n = self.size
        mid = self.rng.randrange(1, n - 1)
        path = [start]
        r, c = start
        while c < mid:
            c += 1; path.append((r, c))
        while r != goal[0]:
            r += 1 if goal[0] > r else -1
            path.append((r, c))
        while c < n - 1:
            c += 1; path.append((r, c))
        return path

    def generate(self):
        n = self.size
        start, goal = self.source, self.exit
        path = self.base_path(start, goal)
        # Grow the route by bumping edges outward into free cells (always terminates).
        target = min(n * n * 3 // 5, self.rng.randint(int(n * 1.5), int(n * 2.2)))
        for _ in range(400):
            if len(path) >= target:
                break
            i = self.rng.randrange(len(path) - 1)
            (r1, c1), (r2, c2) = path[i], path[i + 1]
            dr, dc = (r2 - r1), (c2 - c1)
            side = self.rng.choice((-1, 1))
            q1 = (r1 + dc * side, c1 + dr * side)
            q2 = (r2 + dc * side, c2 + dr * side)
            used = set(path)
            if all(mask_in_bounds(q, n) and q not in used for q in (q1, q2)):
                path[i + 1:i + 1] = [q1, q2]
        self.solution = {}
        for i, (r, c) in enumerate(path):
            mask = 0
            if i:
                pr, pc = path[i-1]
                mask |= E if pc > c else W if pc < c else S if pr > r else N
            if i < len(path)-1:
                nr, nc = path[i+1]
                mask |= E if nc > c else W if nc < c else S if nr > r else N
            self.solution[(r, c)] = mask
        # Fill all cells with harmless random pipes; path cells retain required connections.
        for r in range(n):
            for c in range(n):
                self.tiles[r][c] = self.rng.choice(ALL_PIPES)
        for pos, mask in self.solution.items():
            self.tiles[pos[0]][pos[1]] = mask
        # Scramble path tiles only; unrelated tiles are decorative decoys.
        # Retry so a board never starts already solved.
        movable = [pos for pos in self.solution if pos not in (self.source, self.exit)]
        for _ in range(50):
            self.scramble = {}
            for pos in movable:
                turns = self.rng.randint(1, 3)
                self.tiles[pos[0]][pos[1]] = rotate_mask(self.solution[pos], turns)
                self.scramble[pos] = turns
            if not self.won():
                break
        else:
            if movable:  # fallback: force one tile out of place
                pos = movable[0]
                turns = 1 if self.solution[pos] in STRAIGHT else self.rng.randint(1, 3)
                self.tiles[pos[0]][pos[1]] = rotate_mask(self.solution[pos], turns)

    def rotate(self, pos):
        r, c = pos
        if pos in (self.source, self.exit):
            return False
        self.tiles[r][c] = rotate_mask(self.tiles[r][c])
        return True

    def reachable(self):
        """BFS following only reciprocal pipe openings; returns reached cells."""
        q = deque([self.source])
        seen = {self.source}
        while q:
            r, c = q.popleft()
            mask = self.tiles[r][c]
            for bit, dr, dc, opposite in DIRS:
                nr, nc = r + dr, c + dc
                if not (mask & bit) or not (0 <= nr < self.size and 0 <= nc < self.size):
                    continue
                if not (self.tiles[nr][nc] & opposite):
                    continue
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt); q.append(nxt)
        return seen

    def won(self):
        return self.exit in self.reachable()

    def route(self):
        """Return one reciprocal route source-to-exit for laser rendering."""
        q = deque([self.source])
        parent = {self.source: None}
        while q:
            pos = q.popleft()
            if pos == self.exit:
                out = []
                while pos is not None:
                    out.append(pos); pos = parent[pos]
                return list(reversed(out))
            r, c = pos
            for bit, dr, dc, opposite in DIRS:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if (mask_in_bounds(nxt, self.size) and self.tiles[r][c] & bit
                        and self.tiles[nr][nc] & opposite and nxt not in parent):
                    parent[nxt] = pos; q.append(nxt)
        return []


def mask_in_bounds(pos, size):
    return 0 <= pos[0] < size and 0 <= pos[1] < size


MODEL_FILE = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
CURSOR_TIMEOUT = 0.4   # seconds without a detection before the fingertip cursor disappears
MARGIN = 0.08          # ignore camera edges so every tile is reachable by hand


class HandTracker:
    """Webcam fingertip tracker running on a worker thread.

    Camera open, model download, inference and frame reads never block the
    game loop. Any failure just changes `status`; mouse/keyboard keep working.
    """

    def __init__(self, enabled=True):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.want_enabled = enabled
        self.available = cv2 is not None and mp is not None
        self._norm = None          # smoothed (x, y) in 0..1, mirrored
        self._stamp = 0.0
        self._status = "Webcam off — mouse controls ready"
        self.thread = None
        if not self.available:
            self._status = "OpenCV/MediaPipe not installed — use mouse"
            self.want_enabled = False
            return
        self.thread = threading.Thread(target=self._run, name="hand-tracker", daemon=True)
        self.thread.start()

    # ---- thread-safe accessors -------------------------------------------------
    @property
    def status(self):
        with self.lock:
            return self._status

    def _set_status(self, text):
        with self.lock:
            self._status = text

    @property
    def enabled(self):
        return self.available and self.want_enabled

    @property
    def cursor(self):
        with self.lock:
            if not self.want_enabled or self._norm is None:
                return None
            if time.time() - self._stamp > CURSOR_TIMEOUT:
                return None
            x, y = self._norm
        x = min(1.0, max(0.0, (x - MARGIN) / (1 - 2 * MARGIN)))
        y = min(1.0, max(0.0, (y - MARGIN) / (1 - 2 * MARGIN)))
        return int(x * (WIDTH - 1)), int(y * (HEIGHT - 1))

    def update(self):
        return self.cursor

    def toggle(self):
        if not self.available:
            return False
        self.want_enabled = not self.want_enabled
        with self.lock:
            self._norm = None
        if not self.want_enabled:
            self._set_status("Webcam paused — mouse controls ready")
        return self.want_enabled

    def close(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.5)

    # ---- worker ----------------------------------------------------------------
    def _open_camera(self):
        if sys.platform.startswith("win"):
            backends = [getattr(cv2, "CAP_DSHOW", 0), getattr(cv2, "CAP_MSMF", 0), 0]
        else:
            backends = [0]
        for index in (0, 1):
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        return cap
                    cap.release()
                except Exception:
                    pass
        return None

    def _ensure_model(self):
        if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 1_000_000:
            return True
        self._set_status("Downloading hand model (one time)...")
        tmp = MODEL_FILE.with_suffix(".part")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=20) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            tmp.replace(MODEL_FILE)
            return True
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            return False

    def _make_detector(self):
        """Return (detect(rgb, ts_ms) -> (x, y) | None, close()) for the installed MediaPipe."""
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.55, min_tracking_confidence=0.5)

            def detect(rgb, ts_ms):
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0].landmark[8]
                    return lm.x, lm.y
                return None
            return detect, hands.close

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        if not self._ensure_model():
            raise RuntimeError("model")
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_FILE)),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
        landmarker = vision.HandLandmarker.create_from_options(options)

        def detect(rgb, ts_ms):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(image, ts_ms)
            if res.hand_landmarks:
                lm = res.hand_landmarks[0][8]
                return lm.x, lm.y
            return None
        return detect, landmarker.close

    def _give_up(self, message):
        self._set_status(message)
        self.want_enabled = False
        self.available = False

    def _run(self):
        cap = None
        detect = close_detector = None
        failures = 0
        last_ts = 0
        smooth = None
        try:
            while not self.stop_event.is_set():
                if not self.want_enabled:
                    if cap is not None:         # turn the camera light off while disabled
                        cap.release()
                        cap = None
                    smooth = None
                    self.stop_event.wait(0.1)
                    continue
                if detect is None:
                    self._set_status("Loading hand tracking...")
                    try:
                        detect, close_detector = self._make_detector()
                    except Exception as exc:
                        if str(exc) == "model":
                            self._give_up("Hand model missing (offline?) — see README. Mouse works")
                        else:
                            self._give_up("MediaPipe failed to start — mouse works")
                        return
                if cap is None:
                    self._set_status("Opening camera...")
                    cap = self._open_camera()
                    if cap is None:
                        self._give_up("No camera found / access denied — use mouse")
                        return
                    failures = 0
                    self._set_status("Webcam active — hold fingertip on a tile")
                try:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        failures += 1
                        if failures > 60:
                            self._give_up("Camera stopped responding — mouse still works")
                            return
                        self.stop_event.wait(0.03)
                        continue
                    failures = 0
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_ts = max(last_ts + 1, int(time.monotonic() * 1000))
                    found = detect(rgb, last_ts)
                except Exception:
                    self._set_status("Hand tracking error — mouse still works")
                    self.stop_event.wait(0.2)
                    continue
                if found:
                    smooth = found if smooth is None else (
                        smooth[0] * 0.6 + found[0] * 0.4, smooth[1] * 0.6 + found[1] * 0.4)
                    with self.lock:
                        self._norm = smooth
                        self._stamp = time.time()
                        self._status = "Hand detected — dwell to rotate"
                else:
                    with self.lock:
                        self._status = "No hand detected — show your hand or use mouse"
                self.stop_event.wait(0.015)  # cap at ~30-60 inferences/s
        finally:
            if cap is not None:
                cap.release()
            if close_detector is not None:
                try:
                    close_detector()
                except Exception:
                    pass




class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("LASER LOGIC // Neon Pipe Arcade")
        try:
            # SCALED keeps the 1100x760 layout and maps mouse coords on any screen size.
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED | pygame.RESIZABLE)
        except pygame.error:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = self.make_font(20)
        self.small = self.make_font(15)
        self.medium = self.make_font(27, True)
        self.large = self.make_font(54, True)
        self.state = "MENU"
        self.buttons = []
        self.glow_cache = {}
        self.background = None
        self.selected = (0, 0)
        self.level_points = self.level_moves = 0
        self.level_time = 0.0
        self.tracker = HandTracker()
        self.score = 0
        self.best = self.load_best()
        self.level = 1
        self.combo = 0
        self.moves = 0
        self.time_left = 60.0
        self.feedback = ""
        self.feedback_until = 0
        self.particles = []
        self.hover = None
        self.dwell_pos = None
        self.dwell_start = 0
        self.dwell_fired = False
        self.last_rotate = 0
        self.puzzle = None
        self.grid_rect = pygame.Rect(0, 0, 0, 0)
        self.tile_size = 0
        self.new_level()
        self.state = "MENU"  # new_level() starts play; the player starts from the menu

    @staticmethod
    def make_font(size, bold=False):
        try:
            return pygame.font.SysFont("consolas,couriernew,dejavusansmono,monospace", size, bold=bold)
        except Exception:
            return pygame.font.Font(None, int(size * 1.2))

    def load_best(self):
        try:
            return max(0, int(json.loads(DATA_FILE.read_text(encoding="utf-8")).get("best", 0)))
        except (OSError, ValueError, TypeError, AttributeError):
            return 0

    def save_best(self):
        try:
            tmp = DATA_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps({"best": self.best}, indent=2), encoding="utf-8")
            tmp.replace(DATA_FILE)
        except OSError:
            pass  # read-only folder: keep playing, best score just isn't persisted

    def new_level(self):
        size = min(5 + (self.level - 1)//2, 7)
        self.puzzle = Puzzle(size)
        self.layout()
        self.selected = self.puzzle.source
        self.moves = 0
        self.time_left = min(60 + (self.level-1)*2, 75)
        self.dwell_pos = None
        self.dwell_fired = False
        self.state = "PLAYING"

    def reset_run(self):
        self.score = 0; self.level = 1; self.combo = 0
        self.new_level()

    def rotate_at(self, pos):
        if self.state != "PLAYING" or not pos:
            return
        if self.puzzle.rotate(pos):
            self.moves += 1
            self.feedback = "TILE ROTATED  +1 MOVE"
            self.feedback_until = time.time() + 0.65
            self.spawn_sparks(self.tile_center(pos), CYAN, 5)
            if self.puzzle.won():
                self.complete_level()

    def complete_level(self):
        if self.state != "PLAYING":
            return
        time_bonus = int(max(0, self.time_left) * 10)
        efficiency = max(0, 160 - self.moves * 8)
        base = 250 + time_bonus + efficiency
        self.combo += 1
        earned = int(base * (1 + min(self.combo-1, 5)*0.15))
        self.score += earned
        self.best = max(self.best, self.score)
        self.save_best()
        self.level_points = earned
        self.level_time = self.time_left
        self.level_moves = self.moves
        self.state = "LEVEL_COMPLETE"
        self.spawn_sparks((WIDTH//2, HEIGHT//2), GREEN, 50)

    def game_over(self):
        if self.state != "PLAYING":
            return
        self.state = "GAME_OVER"
        self.combo = 0
        self.best = max(self.best, self.score)
        self.save_best()

    def spawn_sparks(self, center, color, count):
        for _ in range(count):
            a = random.random()*math.tau
            speed = random.uniform(45, 230)
            self.particles.append([center[0], center[1], math.cos(a)*speed,
                                   math.sin(a)*speed, random.uniform(.25, .8), color])

    def update_particles(self, dt):
        for p in self.particles:
            p[0] += p[2]*dt; p[1] += p[3]*dt; p[4] -= dt
        self.particles = [p for p in self.particles if p[4] > 0]

    def tile_center(self, pos):
        r, c = pos
        return (self.grid_rect.x + c*self.tile_size + self.tile_size//2,
                self.grid_rect.y + r*self.tile_size + self.tile_size//2)

    def tile_at(self, point):
        if not point or not self.grid_rect.collidepoint(point):
            return None
        c = (point[0]-self.grid_rect.x)//self.tile_size
        r = (point[1]-self.grid_rect.y)//self.tile_size
        if 0 <= r < self.puzzle.size and 0 <= c < self.puzzle.size:
            return (int(r), int(c))
        return None

    def draw_text(self, text, pos, font=None, color=WHITE, center=False):
        surf = (font or self.font).render(str(text), True, color)
        rect = surf.get_rect()
        if center: rect.center = pos
        else: rect.topleft = pos
        self.screen.blit(surf, rect)
        return rect

    def panel(self, rect, fill=PANEL, border=(39, 73, 115), radius=16):
        pygame.draw.rect(self.screen, fill, rect, border_radius=radius)
        pygame.draw.rect(self.screen, border, rect, width=1, border_radius=radius)

    def draw_background(self, now):
        if self.background is None:  # static, so build it once
            self.background = pygame.Surface((WIDTH, HEIGHT))
            self.background.fill(BG)
            for x in range(0, WIDTH, 40):
                pygame.draw.line(self.background, (12, 19, 40), (x, 0), (x, HEIGHT), 1)
            for y in range(0, HEIGHT, 40):
                pygame.draw.line(self.background, (12, 19, 40), (0, y), (WIDTH, y), 1)
        self.screen.blit(self.background, (0, 0))

    def draw_button(self, rect, label, accent=CYAN, active=False):
        fill = (25, 47, 76) if active else (17, 29, 56)
        pygame.draw.rect(self.screen, fill, rect, border_radius=11)
        pygame.draw.rect(self.screen, accent, rect, 2, border_radius=11)
        self.draw_text(label, rect.center, self.font, WHITE, True)

    def draw_pipe(self, center, mask, size, active=False, source=False, exit_tile=False, reached=False, now=0):
        """Draw a pipe whose arms run to the tile edge, so connected neighbours visibly join.

        `size` is the full cell size. Dim pipes are empty; bright pipes carry the laser.
        """
        x, y = center
        half = size // 2
        width = max(6, size // 6)
        lit = reached or source
        if source:
            color = BLUE
        elif exit_tile:
            color = GREEN
        else:
            color = CYAN if lit else PIPE_DIM
        # DIRS holds (bit, row step, column step): columns are x, rows are y.
        arms = [(x + dc * half, y + dr * half) for bit, dr, dc, _ in DIRS if mask & bit]
        if lit:
            key = (size, mask)
            glow = self.glow_cache.get(key)
            if glow is None:
                glow = pygame.Surface((size, size), pygame.SRCALPHA)
                for bit, dr, dc, _ in DIRS:
                    if mask & bit:
                        pygame.draw.line(glow, CYAN + (40,), (half, half),
                                         (half + dc * half, half + dr * half), width * 2 + 4)
                pygame.draw.circle(glow, CYAN + (40,), (half, half), width + 2)
                self.glow_cache[key] = glow
            self.screen.blit(glow, (x - half, y - half))
        for end in arms:
            pygame.draw.line(self.screen, PIPE_EDGE, (x, y), end, width + 6)
        pygame.draw.circle(self.screen, PIPE_EDGE, (x, y), (width + 6) // 2)
        for end in arms:
            pygame.draw.line(self.screen, color, (x, y), end, width)
        pygame.draw.circle(self.screen, color, (x, y), width // 2)
        if lit:  # bright core = the laser travelling inside the pipe
            for end in arms:
                pygame.draw.line(self.screen, WHITE, (x, y), end, max(2, width // 4))
        if source or exit_tile:
            r = max(10, int(size * .2))
            pygame.draw.circle(self.screen, color if (source or reached) else PANEL, (x, y), r)
            pygame.draw.circle(self.screen, color, (x, y), r, 2)
            self.draw_text("S" if source else "E", (x, y), self.small,
                           BG if (source or reached) else WHITE, True)

    def draw_laser(self, now):
        route = self.puzzle.route()
        if len(route) > 1:
            pts = [self.tile_center(p) for p in route]
            # travelling pulse
            total = len(pts)-1
            t = (now*1.25) % max(1, total)
            idx = min(int(t), total-1)
            frac = t-idx
            px = int(pts[idx][0]*(1-frac)+pts[idx+1][0]*frac)
            py = int(pts[idx][1]*(1-frac)+pts[idx+1][1]*frac)
            pygame.draw.circle(self.screen, WHITE, (px, py), 6)
            pygame.draw.circle(self.screen, CYAN, (px, py), 11, 2)

    def draw_hud(self, now):
        x = 20
        self.draw_text("LASER LOGIC", (x, 10), self.medium, CYAN)
        x = 280
        for label, value in [("SCORE", f"{self.score:06d}"), ("LVL", f"{self.level}"),
                             ("TIME", f"{max(0,int(self.time_left)):02d}s"), ("MOVES", self.moves)]:
            self.draw_text(label, (x, 10), self.small, MUTED)
            self.draw_text(str(value), (x, 28), self.font, RED if label=="TIME" and self.time_left < 12 else CYAN)
            x += 180

    def layout(self):
        """Board geometry; set on every new level so clicks map correctly before the first draw."""
        n = self.puzzle.size
        self.tile_size = min(88, 480 // n)
        board_w = n * self.tile_size
        self.grid_rect = pygame.Rect((WIDTH - board_w) // 2, 80, board_w, board_w)

    def draw_game(self, now):
        self.draw_hud(now)
        n = self.puzzle.size
        self.layout()
        self.panel(self.grid_rect.inflate(18, 18), (10, 18, 39), (31, 66, 101), 18)
        reached = self.puzzle.reachable()
        for r in range(n):
            for c in range(n):
                pos = (r, c)
                rect = pygame.Rect(self.grid_rect.x+c*self.tile_size+3,
                                   self.grid_rect.y+r*self.tile_size+3,
                                   self.tile_size-6, self.tile_size-6)
                pygame.draw.rect(self.screen, (14, 25, 49), rect, border_radius=12)
                pygame.draw.rect(self.screen, (31, 53, 84), rect, 1, border_radius=12)
                active = pos == self.selected
                if active:
                    pygame.draw.rect(self.screen, (54, 32, 85), rect, border_radius=12)
                    pygame.draw.rect(self.screen, MAGENTA, rect, 2, border_radius=12)
                mask = self.puzzle.tiles[r][c]
                self.draw_pipe(self.tile_center(pos), mask, self.tile_size, active,
                               pos == self.puzzle.source, pos == self.puzzle.exit,
                               pos in reached, now)
        self.draw_laser(now)  # travelling pulse once the circuit is complete
        # dwell progress ring
        if self.dwell_pos and not self.dwell_fired:
            progress = min(1.0, (time.time()-self.dwell_start)/DWELL_SECONDS)
            cx, cy = self.tile_center(self.dwell_pos)
            pygame.draw.circle(self.screen, (18, 37, 60), (cx, cy), self.tile_size//2-5, 4)
            pygame.draw.arc(self.screen, MAGENTA, pygame.Rect(cx-self.tile_size//2+5, cy-self.tile_size//2+5,
                            self.tile_size-10, self.tile_size-10), -math.pi/2,
                            -math.pi/2+math.tau*progress, 5)
        hint_y = self.grid_rect.bottom + 34
        self.draw_text("Turn pipes so the bright laser path runs from S to E", (WIDTH//2, hint_y), self.font, WHITE, True)
        self.draw_text("Click a pipe to rotate   •   Esc pause   •   R restart   •   C camera",
                       (WIDTH//2, hint_y + 28), self.small, MUTED, True)
        self.draw_text(self.tracker.status, (WIDTH//2, HEIGHT - 18), self.small,
                       GREEN if self.tracker.enabled else MUTED, True)
        # webcam cursor
        tip = self.tracker.cursor
        if tip:
            pygame.draw.circle(self.screen, (0, 0, 0), tip, 12)
            pygame.draw.circle(self.screen, CYAN, tip, 10, 2)
            pygame.draw.circle(self.screen, WHITE, tip, 3)

    def draw_particles(self):
        for x, y, vx, vy, life, color in self.particles:
            pygame.draw.circle(self.screen, color, (int(x), int(y)), max(1, int(life*4)))

    def overlay(self, title, subtitle, lines, buttons, accent=CYAN):
        shade = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        shade.fill((3, 7, 19, 205))
        self.screen.blit(shade, (0, 0))
        box = pygame.Rect(280, 180, 540, 380)
        self.panel(box, (12, 22, 47), accent, 22)
        self.draw_text(title, (WIDTH//2, 215), self.large, accent, True)
        self.draw_text(subtitle, (WIDTH//2, 265), self.font, WHITE, True)
        y = 310
        for line in lines:
            self.draw_text(line, (WIDTH//2, y), self.small, MUTED, True); y += 25
        self.buttons = []
        by = max(470, y + 15)
        for i, (label, action) in enumerate(buttons):
            rect = pygame.Rect(WIDTH//2-110, by+i*48, 220, 40)
            self.draw_button(rect, label, accent, True)
            self.buttons.append((rect, action))

    def draw_menu(self):
        cx = WIDTH // 2
        self.draw_text("LASER LOGIC", (cx, 62), self.large, CYAN, True)
        self.draw_text("Rotate the pipes. Guide the laser to the exit before time runs out.",
                       (cx, 108), self.font, WHITE, True)

        card = pygame.Rect(210, 140, 680, 410)
        self.panel(card)
        self.draw_text("HOW TO PLAY", (cx, 172), self.medium, CYAN, True)
        steps = [
            ("1", "The laser starts at the blue S tile.", BLUE),
            ("2", "Click a pipe to turn it 90° clockwise.", CYAN),
            ("3", "Bright pipes carry the laser. Make the bright path grow.", CYAN),
            ("4", "Reach the green E tile to clear the level.", GREEN),
            ("5", "Beat the 60s clock. Fewer moves = more points.", MAGENTA),
        ]
        y = 205
        for num, text, color in steps:
            pygame.draw.circle(self.screen, color, (card.x + 60, y + 11), 14, 2)
            self.draw_text(num, (card.x + 60, y + 11), self.small, WHITE, True)
            self.draw_text(text, (card.x + 92, y), self.font, WHITE)
            y += 38

        # Legend: the three tile types the player will see.
        pygame.draw.line(self.screen, (39, 73, 115), (card.x + 40, 400), (card.right - 40, 400), 1)
        legend = [("Source", E, True, False, False), ("Empty pipe", E | S, False, False, False),
                  ("Laser inside", E | W, False, False, True), ("Exit", W, False, True, False)]
        for i, (label, mask, src, ext, lit) in enumerate(legend):
            lx = card.x + 130 + i * 140
            self.draw_pipe((lx, 445), mask, 56, source=src, exit_tile=ext, reached=lit)
            self.draw_text(label, (lx, 490), self.small, MUTED, True)
        self.draw_text("Keyboard: arrows + Enter     Webcam: hold a fingertip on a tile",
                       (cx, card.bottom - 26), self.small, MUTED, True)

        rect = pygame.Rect(cx - 125, 575, 250, 54)
        self.draw_button(rect, "START GAME", CYAN, True)
        self.buttons.append((rect, "start"))
        self.draw_text("or press Enter", (cx, 648), self.small, MUTED, True)
        self.draw_text(f"Best score: {self.best}", (cx, 688), self.font, WHITE, True)
        self.draw_text(self.tracker.status, (cx, HEIGHT - 24), self.small,
                       GREEN if self.tracker.enabled else MUTED, True)

    def draw(self):
        now = time.time()
        self.buttons = []  # rebuilt by whichever screen draws buttons this frame
        self.draw_background(now)
        if self.state == "MENU":
            self.draw_menu()
        elif self.state in ("PLAYING", "PAUSED"):
            self.draw_game(now)
            if self.state == "PAUSED":
                self.overlay("PAUSED", "Timer stopped",
                             ["Click pipes to rotate them. Connect S to E."],
                             [("RESUME", "resume"), ("MAIN MENU", "menu")], PURPLE)
        elif self.state == "LEVEL_COMPLETE":
            self.draw_game(now)
            self.overlay("LEVEL CLEAR!", f"+{self.level_points} points",
                         [f"{int(self.level_time)}s left   •   {self.level_moves} moves   •   combo x{self.combo}"],
                         [("NEXT LEVEL", "next"), ("MAIN MENU", "menu")], GREEN)
        elif self.state == "GAME_OVER":
            best = "New best score!" if self.score >= self.best and self.score > 0 else f"Best: {self.best}"
            self.overlay("TIME'S UP", f"Score: {self.score}",
                         [f"Reached level {self.level}   •   {best}"],
                         [("PLAY AGAIN", "again"), ("MAIN MENU", "menu")], MAGENTA)
        self.draw_particles()
        pygame.display.flip()

    def handle_action(self, action):
        if action == "start" or action == "again":
            self.reset_run()
        elif action == "next":
            self.level += 1
            self.new_level()
        elif action == "resume":
            self.state = "PLAYING"
        elif action == "menu":
            self.state = "MENU"

    def move_selection(self, dr, dc):
        n = self.puzzle.size
        r, c = self.selected
        self.selected = ((r + dr) % n, (c + dc) % n)

    def update_dwell(self, now, hand_cursor):
        """One rotation per dwell; the fingertip must leave the tile to re-arm."""
        if hand_cursor is not None and self.hover is not None:
            if self.hover != self.dwell_pos:
                self.dwell_pos = self.hover
                self.dwell_start = now
                self.dwell_fired = False
            elif not self.dwell_fired and now - self.dwell_start >= DWELL_SECONDS:
                self.dwell_fired = True
                self.rotate_at(self.hover)
        else:
            self.dwell_pos = None
            self.dwell_fired = False

    def handle_key(self, key):
        playing = self.state == "PLAYING"
        if key in (pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_p):
            if playing:
                self.state = "PAUSED"
            elif self.state == "PAUSED":
                self.state = "PLAYING"
            elif self.state == "MENU" and key == pygame.K_ESCAPE:
                return False
        elif key == pygame.K_c:
            self.tracker.toggle()
        elif key == pygame.K_r and self.state in ("PLAYING", "PAUSED"):
            self.new_level()
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.state == "MENU":
                self.reset_run()
            elif playing:
                self.rotate_at(self.selected)
            elif self.state == "LEVEL_COMPLETE":
                self.handle_action("next")
            elif self.state == "GAME_OVER":
                self.handle_action("again")
            elif self.state == "PAUSED":
                self.state = "PLAYING"
        elif playing:
            moves = {pygame.K_UP: (-1, 0), pygame.K_w: (-1, 0), pygame.K_DOWN: (1, 0),
                     pygame.K_s: (1, 0), pygame.K_LEFT: (0, -1), pygame.K_a: (0, -1),
                     pygame.K_RIGHT: (0, 1), pygame.K_d: (0, 1)}
            if key in moves:
                self.move_selection(*moves[key])
        return True

    def run(self):
        running = True
        try:
            while running:
                dt = min(self.clock.tick(FPS)/1000.0, 0.05)
                now = time.time()
                self.update_particles(dt)
                playing = self.state == "PLAYING"
                hand = self.tracker.cursor if playing else None
                if playing:
                    # Hand hover drives the dwell; the mouse updates the selection only when it
                    # moves, so it doesn't override arrow-key selection while resting on the board.
                    self.hover = self.tile_at(hand)
                    if self.hover is not None:
                        self.selected = self.hover
                    self.time_left -= dt
                    if self.time_left <= 0:
                        self.time_left = 0
                        self.game_over()
                    self.update_dwell(now, hand)
                else:
                    self.hover = None
                    self.dwell_pos = None
                    self.dwell_fired = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if not self.handle_key(event.key):
                            running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        handled = False
                        for rect, action in self.buttons:
                            if rect.collidepoint(event.pos):
                                self.handle_action(action); handled = True; break
                        if not handled and self.state == "PLAYING":
                            tile = self.tile_at(event.pos)
                            if tile is not None:
                                self.selected = tile
                            self.rotate_at(tile)
                    elif event.type == pygame.MOUSEMOTION and self.state == "PLAYING":
                        tile = self.tile_at(event.pos)
                        if tile is not None:
                            self.selected = tile
                self.hand_cursor = hand
                self.draw()
        finally:
            self.tracker.close()
            pygame.quit()


def self_test():
    """Quick deterministic logic checks; run with: python main.py --test"""
    assert rotate_mask(N, 1) == E
    assert rotate_mask(E, 1) == S
    assert rotate_mask(S, 1) == W
    assert rotate_mask(W, 1) == N
    assert connected(E, W, E)
    assert not connected(E, N, E)
    p = Puzzle(5, seed=123)
    # Restore its generated solution orientations, then prove the route.
    for pos, mask in p.solution.items():
        p.tiles[pos[0]][pos[1]] = mask
    assert p.won(), "valid solution should win"
    # Break the first interior tile's orientation; the specific seed has an interior path tile.
    broken = next(pos for pos in p.solution if pos not in (p.source, p.exit))
    p.tiles[broken[0]][broken[1]] = rotate_mask(p.tiles[broken[0]][broken[1]], 1)
    # A single turn could still preserve one branch in a route; verify no false-positive via a known disconnected board.
    p.tiles = [[N for _ in range(p.size)] for _ in range(p.size)]
    p.tiles[p.source[0]][p.source[1]] = E
    p.tiles[p.exit[0]][p.exit[1]] = W
    assert not p.won(), "mismatched route must not win"
    # Four quarter-turns restore every pipe; straights/corners stay straights/corners.
    for pipe in ALL_PIPES:
        assert rotate_mask(pipe, 4) == pipe
        assert (rotate_mask(pipe) in STRAIGHT) == (pipe in STRAIGHT)
    # One-way opening (A opens east, B does not open west) must not connect.
    p = Puzzle(5, seed=1)
    p.tiles = [[E | W] * p.size for _ in range(p.size)]
    sr = p.source[0]
    p.tiles[sr][1] = N | S
    assert (sr, 1) not in p.reachable(), "one-way opening must not connect"
    # Fixed tiles never rotate.
    before = p.tiles[sr][0]
    assert not p.rotate(p.source) and p.tiles[sr][0] == before
    for seed in range(300):
        board = Puzzle(5 + seed % 3, seed=seed)
        assert not board.won(), f"board {seed} starts already solved"
        path = list(board.solution)
        assert path[0] == board.source and path[-1] == board.exit
        for pos, mask in board.solution.items():
            assert mask_in_bounds(pos, board.size)
            if pos not in (board.source, board.exit):
                assert mask in ALL_PIPES, f"unsupported pipe shape in board {seed}"
            board.tiles[pos[0]][pos[1]] = mask
        assert board.won(), f"generated solution invalid for seed {seed}"
    print("Logic checks passed: rotation, reciprocal connections, fixed tiles, win/loss validation, "
          "300 solvable boards (none pre-solved).")


def main():
    if "--test" in sys.argv:
        self_test()
        return 0
    if pygame is None:
        print("Pygame is not installed. Run:  python -m pip install -r requirements.txt")
        return 1
    try:
        Game().run()
    except pygame.error as exc:
        print(f"Could not start the display/audio system: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
