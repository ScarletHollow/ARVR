from pathlib import Path

import numpy as np

from core_imu import load_imu_data, run_full_sensor_fusion
from problem3_common import run_tracking_video


def main():
    engine_dir = Path(__file__).resolve().parent
    # Keep alpha conservative here so beta isolates the magnetic yaw correction stage.
    alpha = 0.02
    beta = 0.20

    imu_data = load_imu_data(engine_dir / "IMUData.csv")
    fused_result = run_full_sensor_fusion(imu_data, alpha=alpha, beta=beta)
    final_yaw_deg = float(np.degrees(fused_result["full_fused_eulers"][-1][2]))
    mean_pre_deg = float(np.degrees(np.mean(np.abs(fused_result["pre_yaw_error_angles"]))))
    mean_post_deg = float(np.degrees(np.mean(np.abs(fused_result["post_yaw_error_angles"]))))
    final_pre_deg = float(np.degrees(fused_result["pre_yaw_error_angles"][-1]))
    final_post_deg = float(np.degrees(fused_result["post_yaw_error_angles"][-1]))
    valid_mask = fused_result["mag_valid_mask"]
    # Heading summaries are only meaningful on frames where the horizontal magnetic direction is valid.
    valid_pre_deg = np.rad2deg(np.abs(fused_result["pre_yaw_error_angles"][valid_mask])).astype(float)
    valid_post_deg = np.rad2deg(np.abs(fused_result["post_yaw_error_angles"][valid_mask])).astype(float)
    applied_yaw_correction_deg = beta * valid_pre_deg
    correction_change_deg = np.abs(np.diff(applied_yaw_correction_deg)) if applied_yaw_correction_deg.size > 1 else np.array([0.0])
    p95_post_deg = float(np.percentile(valid_post_deg, 95))
    p95_applied_deg = float(np.percentile(applied_yaw_correction_deg, 95))
    p95_correction_change_deg = float(np.percentile(correction_change_deg, 95))
    print(
        "Problem 4.1 summary | "
        f"alpha={alpha:.2f} | beta={beta:.2f} | "
        f"final_yaw_deg={final_yaw_deg:.6f} | "
        f"mean_heading_error_pre_deg={mean_pre_deg:.6f} | "
        f"mean_heading_error_post_deg={mean_post_deg:.6f} | "
        f"p95_heading_error_post_deg={p95_post_deg:.6f} | "
        f"final_heading_error_pre_deg={final_pre_deg:.6f} | "
        f"final_heading_error_post_deg={final_post_deg:.6f} | "
        f"p95_applied_yaw_correction_deg={p95_applied_deg:.6f} | "
        f"p95_correction_change_deg={p95_correction_change_deg:.6f}"
    )
    run_tracking_video(
        quaternions=fused_result["full_fused_quaternions"],
        times=fused_result["times"],
        title_text=f"Problem 4.1: Full IMU (a={alpha:.2f}, b={beta:.2f})",
        subtitle_text="Mag yaw correction with physics collisions",
        window_title="VR - Problem 4.1: Full IMU Orientation Playback",
        frame_step=4,
    )


if __name__ == "__main__":
    main()
