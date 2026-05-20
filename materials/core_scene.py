import numpy as np


class Transform:
    @staticmethod
    def translate(x, y, z):
        matrix = np.eye(4)
        matrix[0, 3], matrix[1, 3], matrix[2, 3] = x, y, z
        return matrix

    @staticmethod
    def scale(x, y, z):
        matrix = np.eye(4)
        matrix[0, 0], matrix[1, 1], matrix[2, 2] = x, y, z
        return matrix

    @staticmethod
    def rotate_x(angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def rotate_y(angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def rotate_z(angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])


class Entity:
    def __init__(self, obj_path, pos=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(200, 200, 200)):
        self.verts, self.faces = self.load_obj(obj_path)
        self.pos = np.array(pos, dtype=float)
        self.scale_factors = np.array(scale, dtype=float)
        self.color = tuple(int(channel) for channel in color)
        self.rotation_matrix = np.eye(4)

    def load_obj(self, filename):
        vertices = []
        faces = []
        with open(filename) as handle:
            for line in handle.readlines():
                values = line.split()
                if not values:
                    continue
                if values[0] == "v":
                    vertices.append([float(x) for x in values[1:4]])
                elif values[0] == "f":
                    faces.append([int(x.split("/")[0]) - 1 for x in values[1:4]])

        vertices = np.asarray(vertices, dtype=float)
        vmin, vmax = vertices.min(axis=0), vertices.max(axis=0)
        vertices -= (vmin + vmax) / 2.0
        scale = np.max(vmax - vmin)
        if scale > 0.0:
            vertices /= (scale / 2.0)
        return vertices, np.asarray(faces, dtype=int)

    def get_model_matrix(self):
        return Transform.translate(*self.pos) @ self.rotation_matrix @ Transform.scale(*self.scale_factors)


class PhysicsActor:
    def __init__(self, entity, body, spin_rates=(0.0, 0.0, 0.0), hit_color=(255, 255, 255)):
        self.entity = entity
        self.body = body
        self.spin_rates = np.asarray(spin_rates, dtype=float)
        self.base_color = np.asarray(entity.color, dtype=float)
        self.hit_color = np.asarray(hit_color, dtype=float)
        self.flash_frames = 0

    def notify_collision(self, duration_frames=10):
        self.flash_frames = max(self.flash_frames, int(duration_frames))

    def update_visual(self, elapsed_time):
        self.entity.pos = self.body.position.copy()
        rotation = (
            Transform.rotate_z(self.spin_rates[2] * elapsed_time)
            @ Transform.rotate_y(self.spin_rates[1] * elapsed_time)
            @ Transform.rotate_x(self.spin_rates[0] * elapsed_time)
        )
        self.entity.rotation_matrix = rotation

        if self.flash_frames > 0:
            # Briefly brighten bodies after impact so collisions read clearly in the preview scene.
            mix = 0.45 + 0.55 * (self.flash_frames / 10.0)
            color = np.clip(self.base_color * (1.0 - mix) + self.hit_color * mix, 0.0, 255.0)
            self.entity.color = tuple(int(channel) for channel in color)
            self.flash_frames -= 1
        else:
            self.entity.color = tuple(int(channel) for channel in self.base_color)


class Renderer:
    def __init__(self, width=640, height=480):
        self.width, self.height = width, height
        self.image = np.zeros((height, width, 4), dtype=np.uint8)
        self.zbuffer = 1e18 * np.ones((height, width), dtype=float)
        self.coords = np.mgrid[0:height, 0:width].astype(int)

    def clear(self):
        self.image[:] = np.array([18, 20, 26, 255], dtype=np.uint8)
        self.zbuffer.fill(1e18)

    def triangle(self, t_inv, v0, v1, v2, intensity, color):
        xmin = int(max(0, min(v0[0], v1[0], v2[0])))
        xmax = int(min(self.width, max(v0[0], v1[0], v2[0]) + 1))
        ymin = int(max(0, min(v0[1], v1[1], v2[1])))
        ymax = int(min(self.height, max(v0[1], v1[1], v2[1]) + 1))
        if xmin >= xmax or ymin >= ymax:
            return

        pixels = self.coords[:, ymin:ymax, xmin:xmax].reshape(2, -1)
        pixels_xy = np.flipud(pixels)
        barycentric = np.dot(t_inv, np.vstack((pixels_xy, np.ones((1, pixels_xy.shape[1]), dtype=int))))
        mask = np.all(barycentric >= -1e-4, axis=0)
        if not np.any(mask):
            return

        x_coords, y_coords = pixels_xy[0, mask], pixels_xy[1, mask]
        depth = v0[2] * barycentric[0, mask] + v1[2] * barycentric[1, mask] + v2[2] * barycentric[2, mask]
        valid = depth < self.zbuffer[y_coords, x_coords]
        if not np.any(valid):
            return

        x_valid, y_valid, depth_valid = x_coords[valid], y_coords[valid], depth[valid]
        self.zbuffer[y_valid, x_valid] = depth_valid
        self.image[y_valid, x_valid] = [
            (int(color[0]) * int(intensity)) // 255,
            (int(color[1]) * int(intensity)) // 255,
            (int(color[2]) * int(intensity)) // 255,
            255,
        ]

    def render_entity(self, entity, view_proj_mat, vp_mat, light_dir_to_src):
        verts_h = np.c_[entity.verts, np.ones(len(entity.verts))]
        verts_world_h = verts_h @ entity.get_model_matrix().T
        verts_world = verts_world_h[:, :3]
        verts_clip = verts_world_h @ view_proj_mat.T
        w = verts_clip[:, 3]
        valid_verts = w > 1e-6
        if not np.any(valid_verts):
            return

        verts_ndc_h = np.zeros((len(verts_clip), 4), dtype=float)
        verts_ndc_h[valid_verts] = verts_clip[valid_verts, :4] / w[valid_verts, None]
        verts_screen_h = verts_ndc_h @ vp_mat.T
        verts_screen = verts_screen_h[:, :3]

        faces = entity.faces
        face_valid = np.all(valid_verts[faces], axis=1)
        if not np.any(face_valid):
            return

        tris_screen = verts_screen[faces]
        tris_world = verts_world[faces]
        if len(entity.verts) == 4 and len(entity.faces) == 2:
            # Keep the floor bright enough to stay readable against the dark background.
            intensities = np.full(len(faces), 110, dtype=np.uint8)
        else:
            normals = np.cross(tris_world[:, 1] - tris_world[:, 0], tris_world[:, 2] - tris_world[:, 0])
            magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
            magnitudes[magnitudes == 0.0] = 1.0
            normals /= magnitudes
            ambient = 55
            diffuse = np.maximum(0.0, np.dot(normals, light_dir_to_src)) * 200.0
            intensities = np.clip(ambient + diffuse, 0.0, 255.0).astype(np.uint8)

        t_mats = np.transpose(tris_screen, axes=[0, 2, 1]).copy()
        t_mats[:, 2, :] = 1.0
        try:
            t_mats = np.linalg.inv(t_mats)
        except np.linalg.LinAlgError:
            return

        for index in range(len(faces)):
            if not face_valid[index] or intensities[index] <= 0:
                continue
            triangle = tris_screen[index]
            self.triangle(t_mats[index], triangle[0], triangle[1], triangle[2], intensities[index], entity.color)


def lookat(eye, center, up):
    forward = (center - eye) / np.linalg.norm(center - eye)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    true_up = np.cross(side, forward)

    matrix = np.eye(4)
    matrix[0, :3], matrix[1, :3], matrix[2, :3] = side, true_up, -forward
    matrix[0, 3], matrix[1, 3], matrix[2, 3] = -np.dot(side, eye), -np.dot(true_up, eye), np.dot(forward, eye)
    return matrix


def perspective(fov_deg, aspect, near, far):
    scale = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    matrix = np.zeros((4, 4), dtype=float)
    matrix[0, 0], matrix[1, 1] = scale / aspect, scale
    matrix[2, 2], matrix[2, 3] = (far + near) / (near - far), (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def viewport(width, height):
    return np.array([
        [width / 2.0, 0.0, 0.0, width / 2.0],
        [0.0, -height / 2.0, 0.0, height / 2.0],
        [0.0, 0.0, 1000.0, 1000.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
