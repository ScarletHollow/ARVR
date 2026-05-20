import numpy as np

class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @classmethod
    def from_axis_angle(cls, axis, angle_rad):
        axis = np.array(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-10: return cls(1, 0, 0, 0)
        axis /= norm
        s = np.sin(angle_rad / 2.0)
        return cls(np.cos(angle_rad / 2.0), axis[0]*s, axis[1]*s, axis[2]*s)

    def normalize(self):
        mag = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if mag > 1e-10:
            self.w /= mag
            self.x /= mag
            self.y /= mag
            self.z /= mag
        else:
            self.w, self.x, self.y, self.z = 1.0, 0.0, 0.0, 0.0
        return self

    def multiply(self, other):
        """Hamilton product."""
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return Quaternion(w, x, y, z)

    def conjugate(self):
        """Returns the conjugate quaternion."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def rotate_vector(self, v):
        """Rotates a 3D vector v by this quaternion."""
        v_quat = Quaternion(0, v[0], v[1], v[2])
        res_quat = self.multiply(v_quat).multiply(self.conjugate())
        return np.array([res_quat.x, res_quat.y, res_quat.z])

    def sub(self, other):
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def scalar_mul(self, s):
        return Quaternion(self.w * s, self.x * s, self.y * s, self.z * s)

    def to_rotation_matrix(self):
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w,       0.0],
            [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w,       0.0],
            [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y,   0.0],
            [0.0,                   0.0,                 0.0,                 1.0]
        ])

    def to_array(self):
        return np.array([self.w, self.x, self.y, self.z])

    def __repr__(self):
        return f"Quat({self.w:.4f}, {self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


def _coerce_quaternion_array(q):
    if isinstance(q, Quaternion):
        arr = q.to_array()
    else:
        arr = np.asarray(q, dtype=float)

    if arr.shape != (4,):
        raise ValueError("Quaternion input must be a Quaternion or length-4 iterable [w, x, y, z].")
    return arr.astype(float, copy=False)


def euler_to_quat(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ], dtype=float)

    norm = np.linalg.norm(q)
    if norm > 1e-10:
        q /= norm
    return q


def quat_to_euler(q):
    w, x, y, z = _coerce_quaternion_array(q)
    norm = np.linalg.norm([w, x, y, z])
    if norm <= 1e-10:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    else:
        w, x, y, z = (np.array([w, x, y, z], dtype=float) / norm)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return (float(roll), float(pitch), float(yaw))


def quat_conjugate(q):
    w, x, y, z = _coerce_quaternion_array(q)
    return np.array([w, -x, -y, -z], dtype=float)


def quat_multiply(a, b):
    w1, x1, y1, z1 = _coerce_quaternion_array(a)
    w2, x2, y2, z2 = _coerce_quaternion_array(b)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quat_to_axis_angle(q):
    w, x, y, z = _coerce_quaternion_array(q)
    arr = np.array([w, x, y, z], dtype=float)
    norm = np.linalg.norm(arr)
    if norm <= 1e-10:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0

    w, x, y, z = arr / norm
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    if sin_half <= 1e-8 or angle <= 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0
    axis = np.array([x, y, z], dtype=float) / sin_half
    return axis, float(angle)


def rotate_by_quat(q, v):
    w, x, y, z = _coerce_quaternion_array(q)
    quat = Quaternion(w, x, y, z).normalize()
    vec = np.asarray(v, dtype=float)
    if vec.shape != (3,):
        raise ValueError("Vector input must be a length-3 iterable.")
    return quat.rotate_vector(vec)
