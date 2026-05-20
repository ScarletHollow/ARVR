from pathlib import Path
import sys
import time


def _show_title_card(title, subtitle_lines, duration_ms=1600):
    try:
        import cv2
        import numpy as np
    except Exception:
        # Keep a simple console fallback if OpenCV is unavailable.
        print(title)
        for line in subtitle_lines:
            print(line)
        time.sleep(duration_ms / 1000.0)
        return

    card = np.zeros((320, 720, 3), dtype=np.uint8)
    card[:] = (18, 20, 26)
    cv2.putText(card, title, (36, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (240, 240, 240), 2)
    y = 170
    for line in subtitle_lines:
        cv2.putText(card, line, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (205, 205, 205), 1)
        y += 34

    window_name = "AR/VR Runtime Overview"
    cv2.imshow(window_name, card)
    cv2.waitKey(duration_ms)
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)


def main():
    project_dir = Path(__file__).resolve().parent
    engine_dir = project_dir / "materials"

    if not engine_dir.exists():
        raise FileNotFoundError(f"Engine directory not found: {engine_dir}")

    # Runtime entry point stays at the repo root, while the problem scripts live in materials/.
    sys.path.insert(0, str(engine_dir))

    from problem32_runtime import main as run_problem32
    from problem41_runtime import main as run_problem41

    _show_title_card(
        "Runtime Scene 1",
        [
            "Problem 3.2: gyroscope + accelerometer",
            "Center bunny uses complementary fusion",
            "Side bunnies fall and collide as physics scene",
        ],
    )
    run_problem32()

    time.sleep(0.6)

    _show_title_card(
        "Runtime Scene 2",
        [
            "Problem 4.1: gyroscope + accelerometer + magnetometer",
            "Center bunny adds magnetometer yaw correction",
            "Same scene layout for direct comparison",
        ],
    )
    run_problem41()

    _show_title_card(
        "Runtime Complete",
        [
            "Required FAQ runtime sequence finished",
            "Submitted video remains Problem 3.1 followed by Problem 3.2",
        ],
        duration_ms=1800,
    )


if __name__ == "__main__":
    main()
