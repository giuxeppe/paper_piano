# PaperPiano

PaperPiano is a prototype that allows you to play a piano keyboard drawn on paper using a webcam.

The system:
1. automatically detects the keyboard region from an angled view,
2. segments white and black keys,
3. tracks up to 10 fingers using MediaPipe,
4. detects touch events via a state automaton with smoothing,
5. generates real-time piano sound.

---

## Requirements

- Python 3.10+ (Python 3.11 recommended)
- A working webcam
- An active audio output device

Main dependencies:
- `opencv-python`
- `mediapipe`
- `numpy`
- `sounddevice`

---

## Installation

```bash
pip install -r requirements.txt
```

## Running the Program
```bash
python main.py
```

## How to play

We advice to connect a smartphone camera to your computer to facilitate framing the keyboard (IRIUN is a valid external application for this scope).

Draw a simple keyboard on paper, make sure the lines are connected and the black keys are actually black.

Put your camera at a stable 20-30 degrees angle, make sure the keyboard is fully framed (should look like a trapezoid).

Run the program

Have fun!

## Runtime Controls
Inside the video window:

q → quit

r → re-detect keyboard (resets tracker and stops all notes)

s → save a snapshot to OUTPUT_DIR


