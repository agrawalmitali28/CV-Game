# LASER LOGIC — Webcam-Controlled Puzzle Arcade

A local neon pipe-connection puzzle built with Python, Pygame, OpenCV, and MediaPipe. Rotate pipe tiles to connect the blue source to the green exit before the timer runs out. The game remains playable with mouse controls if the camera is unavailable.

## Windows PowerShell setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, you can either run this once in the current PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Or skip activation and run the environment's Python directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

## Controls

- **Mouse click:** rotate a pipe clockwise.
- **Webcam hand tracking:** point your index fingertip at a tile and hold for 0.5 seconds. Leave the tile before triggering another rotation.
- **Arrow keys / WASD + Enter:** keyboard play (select a tile, rotate it).
- **C:** toggle webcam tracking.
- **Space / Esc:** pause or resume.
- **R:** restart the current level.
- **Enter:** start from the menu.
- Use the on-screen buttons to start, continue, replay, or return to the menu.

The webcam image is not shown; fingertip coordinates control the in-game cursor. OpenCV mirrors the captured frame for intuitive movement. If no camera or hand is detected, the status text explains the issue and mouse controls remain available.

## Webcam test

After installing dependencies, run `python main.py`, press **C** if needed, and look for `Hand detected` in the status panel while showing one hand to the camera. If the camera is unavailable, test the game with mouse controls. No custom model training or separate model download is required by this project; MediaPipe's installed package provides its hand-tracking implementation.

## Verification

Run the built-in logic checks:

```powershell
python main.py --test
```

This checks direction rotation, reciprocal connection rules, valid and invalid routes, and solvability of generated boards. It does not simulate physical webcam interaction.

## Files

- `main.py` — game, puzzle generation, BFS validation, webcam tracking, UI, scoring, and high-score persistence.
- `requirements.txt` — runtime dependencies.
- `high_score.json` — created automatically when a score is saved.


## Webcam notes (MediaPipe model)

Newer MediaPipe releases (1.x) no longer include the legacy `mp.solutions` API, so the game uses the Tasks `HandLandmarker`, which needs the `hand_landmarker.task` model file next to `main.py`. It is downloaded automatically (about 8 MB, once) on first run if missing. If you are offline, download it manually from
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
and place it beside `main.py`. Older MediaPipe versions with `mp.solutions` are still supported and need no model file.

Camera setup runs on a background thread, so the game opens instantly. If there is no camera, permission is denied, or MediaPipe fails, the status line says why and mouse/keyboard keep working. Frames are never saved or sent anywhere.

`python main.py --test` runs the built-in logic self-checks.
