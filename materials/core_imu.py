import numpy as np
from core_quaternion import Quaternion, quat_to_axis_angle, quat_to_euler, rotate_by_quat

def load_imu_data(filename):
    """Loads IMU data from CSV."""
    # Column 0: time, 1-3: gyro (deg/s), 4-6: accel, 7-9: mag
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    return data

def _copy_quaternion(q):
    return Quaternion(q.w, q.x, q.y, q.z)

def _gyro_delta_quaternion(omega_rad_s, dt):
    omega = np.asarray(omega_rad_s, dtype=float)
    omega_norm = np.linalg.norm(omega)
    angle = omega_norm * float(dt)
    if omega_norm <= 1e-10 or angle <= 1e-10:
        return Quaternion(1.0, 0.0, 0.0, 0.0)
    axis = omega / omega_norm
    return Quaternion.from_axis_angle(axis, angle)

def _quaternion_series_to_eulers(quaternions):
    return np.array([quat_to_euler(q) for q in quaternions], dtype=float)

def _compute_acc_tilt_axis_angle(q_gyro, acc_sensor):
    acc = np.asarray(acc_sensor, dtype=float)
    acc_norm = np.linalg.norm(acc)
    if acc_norm <= 1e-8 or not np.all(np.isfinite(acc)):
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0

    a_sensor = acc / acc_norm
    # Rotate the measured gravity direction into the world frame before comparing it with world up.
    a_world = rotate_by_quat(q_gyro, a_sensor)
    world_norm = np.linalg.norm(a_world)
    if world_norm <= 1e-8 or not np.all(np.isfinite(a_world)):
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0

    a_world /= world_norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    tilt_axis = np.cross(a_world, world_up)
    axis_norm = np.linalg.norm(tilt_axis)
    theta = float(np.arccos(np.clip(np.dot(a_world, world_up), -1.0, 1.0)))

    if axis_norm <= 1e-8 or theta <= 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0
    return tilt_axis / axis_norm, theta

def _compute_horizontal_mag_direction(q_orientation, mag_sensor):
    mag = np.asarray(mag_sensor, dtype=float)
    mag_norm = np.linalg.norm(mag)
    if mag_norm <= 1e-8 or not np.all(np.isfinite(mag)):
        return None

    mag_world = rotate_by_quat(q_orientation, mag / mag_norm)
    mag_horizontal = np.array([mag_world[0], mag_world[1], 0.0], dtype=float)
    horizontal_norm = np.linalg.norm(mag_horizontal)
    if horizontal_norm <= 1e-8 or not np.all(np.isfinite(mag_horizontal)):
        return None
    return mag_horizontal / horizontal_norm

def _signed_horizontal_angle(from_dir, to_dir):
    return float(np.arctan2(
        from_dir[0] * to_dir[1] - from_dir[1] * to_dir[0],
        from_dir[0] * to_dir[0] + from_dir[1] * to_dir[1],
    ))

def integrate_gyro_dead_reckoning(data):
    """P3.1 dead reckoning using q_delta = axis-angle(omega * dt)."""
    times = data[:, 0]
    gyro = np.deg2rad(data[:, 1:4])

    quaternions = []
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    quaternions.append(_copy_quaternion(q))

    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        q_delta = _gyro_delta_quaternion(gyro[i - 1], dt)
        q = q.multiply(q_delta).normalize()
        quaternions.append(_copy_quaternion(q))

    eulers = _quaternion_series_to_eulers(quaternions)
    return {
        "times": times.copy(),
        "quaternions": quaternions,
        "eulers": eulers,
    }

def compute_acc_tilt_correction(q_gyro, acc_sensor):
    """Returns the full tilt-only correction quaternion from accelerometer data."""
    tilt_axis, theta = _compute_acc_tilt_axis_angle(q_gyro, acc_sensor)
    return Quaternion.from_axis_angle(tilt_axis, theta)

def compute_mag_yaw_correction(q_orientation, mag_sensor, mag_reference_horizontal):
    """Returns the full yaw-only correction quaternion from magnetometer data."""
    if mag_reference_horizontal is None:
        return Quaternion(1.0, 0.0, 0.0, 0.0), None, 0.0

    current_horizontal = _compute_horizontal_mag_direction(q_orientation, mag_sensor)
    if current_horizontal is None:
        return Quaternion(1.0, 0.0, 0.0, 0.0), None, 0.0

    yaw_error = _signed_horizontal_angle(current_horizontal, mag_reference_horizontal)
    return Quaternion.from_axis_angle([0.0, 0.0, 1.0], yaw_error), current_horizontal, yaw_error

def run_complementary_filter(data, alpha=0.02):
    """P3.2-P3.3 gyro integration with accelerometer tilt correction."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1].")

    times = data[:, 0]
    gyro = np.deg2rad(data[:, 1:4])
    acc = data[:, 4:7]

    q_gyro = Quaternion(1.0, 0.0, 0.0, 0.0)
    q_fused = Quaternion(1.0, 0.0, 0.0, 0.0)

    gyro_quats = [_copy_quaternion(q_gyro)]
    fused_quats = []
    correction_quats = []
    scaled_correction_quats = []
    correction_axes = []
    correction_angles = []

    init_axis, init_theta = _compute_acc_tilt_axis_angle(q_fused, acc[0])
    init_scaled = Quaternion.from_axis_angle(init_axis, alpha * init_theta)
    q_fused = init_scaled.multiply(q_fused).normalize()

    fused_quats.append(_copy_quaternion(q_fused))
    correction_quats.append(compute_acc_tilt_correction(Quaternion(1.0, 0.0, 0.0, 0.0), acc[0]))
    scaled_correction_quats.append(_copy_quaternion(init_scaled))
    correction_axes.append(init_axis)
    correction_angles.append(init_theta)

    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        q_delta = _gyro_delta_quaternion(gyro[i - 1], dt)

        q_gyro = q_gyro.multiply(q_delta).normalize()
        gyro_quats.append(_copy_quaternion(q_gyro))

        # Predict with gyro integration, then blend back toward the accelerometer tilt estimate.
        q_pred = q_fused.multiply(q_delta).normalize()
        q_correction = compute_acc_tilt_correction(q_pred, acc[i])
        correction_axis, correction_angle = quat_to_axis_angle(q_correction)
        q_scaled = Quaternion.from_axis_angle(correction_axis, alpha * correction_angle)
        q_fused = q_scaled.multiply(q_pred).normalize()

        fused_quats.append(_copy_quaternion(q_fused))
        correction_quats.append(_copy_quaternion(q_correction))
        scaled_correction_quats.append(_copy_quaternion(q_scaled))
        correction_axes.append(correction_axis)
        correction_angles.append(correction_angle)

    return {
        "times": times.copy(),
        "gyro_quaternions": gyro_quats,
        "gyro_eulers": _quaternion_series_to_eulers(gyro_quats),
        "fused_quaternions": fused_quats,
        "fused_eulers": _quaternion_series_to_eulers(fused_quats),
        "correction_quaternions": correction_quats,
        "scaled_correction_quaternions": scaled_correction_quats,
        "correction_axes": np.array(correction_axes, dtype=float),
        "correction_angles": np.array(correction_angles, dtype=float),
    }

def run_full_sensor_fusion(data, alpha=0.02, beta=0.02):
    """P4.1 gyro + accelerometer + magnetometer complementary fusion."""
    alpha = float(alpha)
    beta = float(beta)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1].")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be within [0, 1].")

    times = data[:, 0]
    gyro = np.deg2rad(data[:, 1:4])
    acc = data[:, 4:7]
    mag = data[:, 7:10]

    q_gyro = Quaternion(1.0, 0.0, 0.0, 0.0)
    q_gyro_acc = Quaternion(1.0, 0.0, 0.0, 0.0)
    q_full = Quaternion(1.0, 0.0, 0.0, 0.0)

    gyro_quats = [_copy_quaternion(q_gyro)]
    gyro_acc_quats = []
    full_quats = []
    yaw_correction_quats = []
    scaled_yaw_correction_quats = []
    pre_yaw_error_angles = []
    post_yaw_error_angles = []
    mag_dirs_before = []
    mag_dirs_after = []
    mag_valid_mask = []

    init_axis, init_theta = _compute_acc_tilt_axis_angle(q_gyro_acc, acc[0])
    init_tilt = Quaternion.from_axis_angle(init_axis, alpha * init_theta)
    q_gyro_acc = init_tilt.multiply(q_gyro_acc).normalize()
    q_full = _copy_quaternion(q_gyro_acc)

    mag_reference_horizontal = _compute_horizontal_mag_direction(q_full, mag[0])
    if mag_reference_horizontal is None:
        mag_reference_horizontal = np.array([1.0, 0.0, 0.0], dtype=float)

    init_yaw_correction, init_mag_dir, init_yaw_error = compute_mag_yaw_correction(
        q_full,
        mag[0],
        mag_reference_horizontal,
    )
    init_scaled_yaw = Quaternion.from_axis_angle([0.0, 0.0, 1.0], beta * init_yaw_error)
    q_full = init_scaled_yaw.multiply(q_full).normalize()

    gyro_acc_quats.append(_copy_quaternion(q_gyro_acc))
    full_quats.append(_copy_quaternion(q_full))
    yaw_correction_quats.append(_copy_quaternion(init_yaw_correction))
    scaled_yaw_correction_quats.append(_copy_quaternion(init_scaled_yaw))
    pre_yaw_error_angles.append(init_yaw_error)

    init_post_mag_dir = _compute_horizontal_mag_direction(q_full, mag[0])
    if init_mag_dir is None or init_post_mag_dir is None:
        mag_dirs_before.append([np.nan, np.nan, np.nan])
        mag_dirs_after.append([np.nan, np.nan, np.nan])
        post_yaw_error_angles.append(0.0)
        mag_valid_mask.append(False)
    else:
        mag_dirs_before.append(init_mag_dir)
        mag_dirs_after.append(init_post_mag_dir)
        post_yaw_error_angles.append(_signed_horizontal_angle(init_post_mag_dir, mag_reference_horizontal))
        mag_valid_mask.append(True)

    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        q_delta = _gyro_delta_quaternion(gyro[i - 1], dt)

        q_gyro = q_gyro.multiply(q_delta).normalize()
        gyro_quats.append(_copy_quaternion(q_gyro))

        q_pred_gyro_acc = q_gyro_acc.multiply(q_delta).normalize()
        q_tilt_correction = compute_acc_tilt_correction(q_pred_gyro_acc, acc[i])
        tilt_axis, tilt_angle = quat_to_axis_angle(q_tilt_correction)
        q_tilt_scaled = Quaternion.from_axis_angle(tilt_axis, alpha * tilt_angle)
        q_gyro_acc = q_tilt_scaled.multiply(q_pred_gyro_acc).normalize()
        gyro_acc_quats.append(_copy_quaternion(q_gyro_acc))

        q_pred_full = q_full.multiply(q_delta).normalize()
        q_full_tilt_correction = compute_acc_tilt_correction(q_pred_full, acc[i])
        full_tilt_axis, full_tilt_angle = quat_to_axis_angle(q_full_tilt_correction)
        q_full_tilt_scaled = Quaternion.from_axis_angle(full_tilt_axis, alpha * full_tilt_angle)
        q_tilt_only = q_full_tilt_scaled.multiply(q_pred_full).normalize()

        # The magnetometer only nudges yaw, leaving the accelerometer to handle roll and pitch.
        q_yaw_correction, mag_dir_before, yaw_error = compute_mag_yaw_correction(
            q_tilt_only,
            mag[i],
            mag_reference_horizontal,
        )
        q_yaw_scaled = Quaternion.from_axis_angle([0.0, 0.0, 1.0], beta * yaw_error)
        q_full = q_yaw_scaled.multiply(q_tilt_only).normalize()

        mag_dir_after = _compute_horizontal_mag_direction(q_full, mag[i])
        if mag_dir_before is None or mag_dir_after is None:
            mag_dirs_before.append([np.nan, np.nan, np.nan])
            mag_dirs_after.append([np.nan, np.nan, np.nan])
            post_yaw_error_angles.append(0.0)
            mag_valid_mask.append(False)
        else:
            mag_dirs_before.append(mag_dir_before)
            mag_dirs_after.append(mag_dir_after)
            post_yaw_error_angles.append(_signed_horizontal_angle(mag_dir_after, mag_reference_horizontal))
            mag_valid_mask.append(True)

        full_quats.append(_copy_quaternion(q_full))
        yaw_correction_quats.append(_copy_quaternion(q_yaw_correction))
        scaled_yaw_correction_quats.append(_copy_quaternion(q_yaw_scaled))
        pre_yaw_error_angles.append(yaw_error)

    return {
        "times": times.copy(),
        "gyro_quaternions": gyro_quats,
        "gyro_eulers": _quaternion_series_to_eulers(gyro_quats),
        "gyro_acc_quaternions": gyro_acc_quats,
        "gyro_acc_eulers": _quaternion_series_to_eulers(gyro_acc_quats),
        "full_fused_quaternions": full_quats,
        "full_fused_eulers": _quaternion_series_to_eulers(full_quats),
        "yaw_correction_quaternions": yaw_correction_quats,
        "scaled_yaw_correction_quaternions": scaled_yaw_correction_quats,
        "pre_yaw_error_angles": np.array(pre_yaw_error_angles, dtype=float),
        "post_yaw_error_angles": np.array(post_yaw_error_angles, dtype=float),
        "mag_dirs_before_correction": np.array(mag_dirs_before, dtype=float),
        "mag_dirs_after_correction": np.array(mag_dirs_after, dtype=float),
        "mag_valid_mask": np.array(mag_valid_mask, dtype=bool),
        "mag_reference_horizontal": np.array(mag_reference_horizontal, dtype=float),
    }

def integrate_gyro(data):
    """Integrates gyroscope data to calculate orientation quaternions."""
    return integrate_gyro_dead_reckoning(data)["quaternions"]
