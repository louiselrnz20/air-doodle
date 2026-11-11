# Air Doodle Agent

Air Doodle Agent is a webcam-based intelligent-agent project. Point with one finger to draw an outline in the air. When you finish, hold a fist or press Space to replace the outline with a transparent picture. The current object classes are flower, house, cat, tree, sun, heart, star, and fish.

## Requirements

- Windows 10 or 11
- A webcam
- A Python version supported by MediaPipe Tasks
- The files and folders from this project kept together

**Important:** extract the entire `Air_Doodle_Source.zip` archive to a regular folder before running it. Do not run `air_doodle_agent.py` from AweZip's preview or temporary folder. The `object_pictures` folder must be beside the Python file.

The extracted layout should look like this:

```text
air-doodle-agent/
  air_doodle_agent.py
  hand_landmarker.task
  learned_shape_templates.npz
  object_pictures/
    flower.png
    house.png
    cat.png
    tree.png
    sun.png
    heart.png
    star.png
    fish.png
```

Install the Python packages from PowerShell in the project folder:

```powershell
py -m pip install -r requirements.txt
```

## Run the camera application

```powershell
py air_doodle_agent.py --camera
```

The camera option is also the default, so `py air_doodle_agent.py` works. If Windows selects the wrong camera, try:

```powershell
py air_doodle_agent.py --camera-index 1
```

## Controls

| Input | Action |
|---|---|
| Point with index finger | Draw an outline |
| Hold a fist for about 0.6 seconds | Finish and classify the outline |
| Space or Enter | Finish immediately |
| Open palm or C | Clear the trace and placed pictures |
| 1–8 | Correct the last picture or label the current trace |
| Q or Esc | Quit |

Number keys map to: 1 flower, 2 house, 3 cat, 4 tree, 5 sun, 6 heart, 7 star, and 8 fish. A correction adds the normalized outline to the matching local example archive so future comparisons can use it.

## Demo and camera-free check

Run the packaged outline round-trip self-check without opening the webcam:

```powershell
py air_doodle_agent.py --demo
```

This check uses the same starter examples that the matcher receives. It checks the contour conversion and label wiring; it is not an accuracy test on new drawings.

## How the agent works

MediaPipe Hand Landmarker supplies hand landmarks. Rule-based pose checks distinguish pointing, a fist, and an open palm. The agent tracks the index fingertip, normalizes a finished trace into a binary mask, and compares it with labeled examples using a symmetric chamfer-style distance. The lowest-distance category selects a picture. The interface allows direct correction when the nearest example is wrong.

The application uses OpenCV, NumPy, and MediaPipe Tasks. It expects `hand_landmarker.task`, `learned_shape_templates.npz`, and the `object_pictures` folder beside the Python file. If the model asset is missing, the app attempts to download it on its first camera run. The supplied PNG pictures have transparent backgrounds.

## Files and data handling

- `air_doodle_agent.py` — camera application and matcher
- `hand_landmarker.task` — hand-landmark model asset
- `learned_shape_templates.npz` — labeled outline examples and saved corrections
- `object_pictures/` — transparent output pictures
- `objects_reference.png` — starter outline sheet used for the original examples

Live camera frames and uncorrected fingertip paths remain in memory and are not saved by the application. Corrected outline masks are written to `learned_shape_templates.npz`. Back up that file before resetting or replacing the project if you want to keep learned corrections.

## Known limits

The matcher always chooses the nearest known class for a sufficiently long trace, so an unrelated outline can receive the wrong picture. The eight-category library is small, the gesture rules may be affected by occlusion or camera angle, and no held-out multi-user accuracy evaluation has been run. The included self-check should not be described as generalization performance.
