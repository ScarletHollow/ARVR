import numpy as np

from core_quaternion import rotate_by_quat


def tilt_error_series_deg(quaternions, accelerometer_samples):
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    errors = []
    for q, acc in zip(quaternions, accelerometer_samples):
        acc = np.asarray(acc, dtype=float)
        acc_norm = np.linalg.norm(acc)
        if acc_norm <= 1e-8 or not np.all(np.isfinite(acc)):
            errors.append(np.nan)
            continue

        # Compare the gravity direction implied by the accelerometer with world up.
        acc_world = rotate_by_quat(q, acc / acc_norm)
        world_norm = np.linalg.norm(acc_world)
        if world_norm <= 1e-8 or not np.all(np.isfinite(acc_world)):
            errors.append(np.nan)
            continue

        acc_world /= world_norm
        errors.append(np.degrees(np.arccos(np.clip(np.dot(acc_world, world_up), -1.0, 1.0))))
    return np.asarray(errors, dtype=float)


def quaternion_delta_series_deg(quaternions_a, quaternions_b):
    deltas = []
    for qa, qb in zip(quaternions_a, quaternions_b):
        # Use the shortest relative rotation between the two orientation trajectories.
        delta = qa.conjugate().multiply(qb).normalize()
        deltas.append(2.0 * np.degrees(np.arccos(np.clip(abs(delta.w), -1.0, 1.0))))
    return np.asarray(deltas, dtype=float)


def finite_mean_p95(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.percentile(values, 95))
