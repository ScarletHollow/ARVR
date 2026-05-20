# AR/VR IMU Tracking And Software Rendering Demo

Python implementation of an AR/VR project covering software
rendering, quaternion-based IMU tracking, complementary sensor fusion, and a
small real-time physics scene. The main runtime demonstrates a tracked bunny
model driven by gyroscope, accelerometer, and magnetometer data.

Coursework mark: 79/100.

## Demo

The submitted visual demo is included here:

```text
demo/problem-3-1-3-2-demo.mp4
```

It compares:

1. Gyroscope-only dead reckoning.
2. Gyroscope + accelerometer complementary filtering.

## Project Highlights

- Hand-built software rendering pipeline for `.obj` models.
- Quaternion math for orientation integration and interpolation.
- IMU dead reckoning from gyroscope samples.
- Complementary filtering with accelerometer tilt correction.
- Magnetometer yaw correction for full IMU sensor fusion.
- Simple rigid-body scene with gravity, drag, floor response, and sphere
  collisions.
- Runtime title cards and OpenCV playback for the required tracking sequences.

## Repository Structure

```text
.
├── render.py                         # Main runtime entry point
├── materials/
│   ├── core_quaternion.py            # Quaternion operations
│   ├── core_imu.py                   # IMU integration and sensor fusion
│   ├── core_physics.py               # Physics update and collision handling
│   ├── core_scene.py                 # Software renderer and scene entities
│   ├── core_metrics.py               # Tracking quality metrics
│   ├── problem31_video.py            # Gyro-only video scene
│   ├── problem32_runtime.py          # Gyro + accelerometer runtime scene
│   ├── problem41_runtime.py          # Full IMU runtime scene
│   ├── problem5_demo.py              # Physics demo
│   ├── IMUData.csv                   # Input IMU sample sequence
│   ├── bunny.obj
│   └── floor.obj
├── demo/
│   └── problem-3-1-3-2-demo.mp4
├── report/
│   └── arvr-report.pdf
├── requirements.txt
└── README.md
```

## Requirements

```powershell
pip install -r requirements.txt
```

Tested with Python 3 and:

- `numpy`
- `opencv-python`

## Running The Runtime Demo

From the repository root:

```powershell
python render.py
```

This launches two runtime scenes:

1. `Problem 3.2`: gyroscope + accelerometer complementary fusion.
2. `Problem 4.1`: gyroscope + accelerometer + magnetometer fusion.

Press `q` to close a scene early and continue to the next one.

## Running Individual Components

```powershell
python materials/problem31_video.py
python materials/problem32_runtime.py --alpha 0.20
python materials/problem41_runtime.py
python materials/problem5_demo.py
```

## Notes

This repository is a cleaned public portfolio version of the AR/VR project.
Generated build artifacts, zip archives, old working versions, private course
materials, and assessment records are not included.
