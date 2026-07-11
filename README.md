# Missiles! AI Pilot

Autonomous PC controller for the Google Play Games version of **Missiles!**.

## Setup

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Copy game PNG assets into:

```text
MissilesAI/assets/templates/
```

Use filenames like `plane1.png`, `missile1.png`, `star1.png`, `shield1.png`, and `boost1.png`.

3. Run:

```powershell
python auto_calibrate.py
```

The launcher will find or launch the game, focus the window, click the red start button, calibrate the joystick, and run the AI pilot.

Press `Ctrl+C` to stop. The joystick is released automatically on exit.

## Notes

- Window detection ignores File Explorer, terminals, VS Code, and Visual Studio.
- Best performance requires sprite PNG templates in `assets/templates`.
- If templates are missing, the bot can only do limited fallback player detection.
