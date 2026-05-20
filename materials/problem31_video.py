from pathlib import Path

import numpy as np

from core_imu import integrate_gyro_dead_reckoning, load_imu_data
from problem3_common import run_problem3_video
from core_metrics import finite_mean_p95, tilt_error_series_deg


def main():
    engine_dir = Path(__file__).resolve().parent
    imu_data = load_imu_data(engine_dir / "IMUData.csv")
    dead = integrate_gyro_dead_reckoning(imu_data)
    tilt_errors_deg = tilt_error_series_deg(dead["quaternions"], imu_data[:, 4:7])
    mean_tilt_deg, p95_tilt_deg = finite_mean_p95(tilt_errors_deg)
    final_yaw_deg = float(np.degrees(dead["eulers"][-1][2]))
    print(
        "Problem 3.1 video summary | "
        f"final_yaw_deg={final_yaw_deg:.6f} | "
        f"mean_tilt_error_deg={mean_tilt_deg:.6f} | "
        f"p95_tilt_error_deg={p95_tilt_deg:.6f}"
    )
    run_problem3_video(
        quaternions=dead["quaternions"],
        times=dead["times"],
        title_text="Problem 3.1: Gyro Only",
        subtitle_text="Center bunny tracked with dead reckoning only",
        window_title="VR Video - Problem 3.1",
        frame_step=4,
    )


if __name__ == "__main__":
    main()
