from pathlib import Path
import argparse

import numpy as np

from core_imu import load_imu_data, run_complementary_filter
from problem3_common import run_problem3_video
from core_metrics import finite_mean_p95, quaternion_delta_series_deg, tilt_error_series_deg


def parse_args():
    parser = argparse.ArgumentParser(description="Problem 3.2 fused IMU playback with configurable alpha.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.20,
        help="Complementary-filter accelerometer gain (default: 0.20).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    engine_dir = Path(__file__).resolve().parent
    alpha = float(args.alpha)

    imu_data = load_imu_data(engine_dir / "IMUData.csv")
    fused = run_complementary_filter(imu_data, alpha=alpha)
    tilt_errors_deg = tilt_error_series_deg(fused["fused_quaternions"], imu_data[:, 4:7])
    mean_tilt_deg, p95_tilt_deg = finite_mean_p95(tilt_errors_deg)
    quat_delta_deg = quaternion_delta_series_deg(
        fused["gyro_quaternions"], fused["fused_quaternions"]
    )
    mean_quat_delta_deg, p95_quat_delta_deg = finite_mean_p95(quat_delta_deg)
    applied_correction_deg = np.degrees(np.asarray(fused["correction_angles"], dtype=float)) * alpha
    mean_applied_correction_deg, p95_applied_correction_deg = finite_mean_p95(applied_correction_deg)
    max_applied_correction_deg = float(np.max(applied_correction_deg))
    final_yaw_deg = float(np.degrees(fused["fused_eulers"][-1][2]))
    print(
        "Problem 3.2 summary | "
        f"alpha={alpha:.2f} | "
        f"final_yaw_deg={final_yaw_deg:.6f} | "
        f"mean_tilt_error_deg={mean_tilt_deg:.6f} | "
        f"p95_tilt_error_deg={p95_tilt_deg:.6f} | "
        f"mean_quat_delta_deg={mean_quat_delta_deg:.6f} | "
        f"p95_quat_delta_deg={p95_quat_delta_deg:.6f} | "
        f"mean_applied_correction_deg={mean_applied_correction_deg:.6f} | "
        f"p95_applied_correction_deg={p95_applied_correction_deg:.6f} | "
        f"max_applied_correction_deg={max_applied_correction_deg:.6f}"
    )
    run_problem3_video(
        quaternions=fused["fused_quaternions"],
        times=fused["times"],
        title_text=f"Problem 3.2: Gyro + Accel (alpha={alpha:.2f})",
        subtitle_text="Center bunny tracked with complementary filter",
        window_title="VR - Problem 3.2: Fused IMU Orientation Playback",
        frame_step=4,
    )


if __name__ == "__main__":
    main()
