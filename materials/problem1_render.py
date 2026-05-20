import numpy as np
import cv2

class Transform:
    @staticmethod
    def translate(x, y, z):
        M = np.eye(4)
        M[0, 3], M[1, 3], M[2, 3] = x, y, z
        return M

    @staticmethod
    def scale(x, y, z):
        M = np.eye(4)
        M[0, 0], M[1, 1], M[2, 2] = x, y, z
        return M

    @staticmethod
    def rotate_x(angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [1, 0,  0, 0],
            [0, c, -s, 0],
            [0, s,  c, 0],
            [0, 0,  0, 1]
        ])

    @staticmethod
    def rotate_y(angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]
        ])

    @staticmethod
    def rotate_z(angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

class Entity:
    def __init__(self, obj_path, pos=(0,0,0), rot=(0,0,0), scale=(1,1,1),
                 color_factor=1.0, cull_backface=True):
        self.verts, self.faces = self.load_obj(obj_path)
        self.pos = np.array(pos, dtype=float)
        self.rot = np.array(rot, dtype=float)
        self.scale_factors = np.array(scale, dtype=float)
        self.color_factor = color_factor
        self.cull_backface = cull_backface

    def load_obj(self, filename):
        V, Vi = [], []
        with open(filename) as f:
            for line in f.readlines():
                values = line.split()
                if not values: continue
                if values[0] == 'v':
                    V.append([float(x) for x in values[1:4]])
                elif values[0] == 'f':
                    Vi.append([int(x.split('/')[0]) - 1 for x in values[1:4]])
        
        V = np.array(V)
        vmin, vmax = V.min(axis=0), V.max(axis=0)
        V -= (vmin + vmax) / 2
        s = np.max(vmax - vmin)
        if s > 0: V /= (s / 2.0)
        return V, np.array(Vi)

    def get_model_matrix(self):
        T = Transform.translate(*self.pos)
        R = Transform.rotate_z(self.rot[2]) @ Transform.rotate_y(self.rot[1]) @ Transform.rotate_x(self.rot[0])
        S = Transform.scale(*self.scale_factors)
        return T @ R @ S

class Renderer:
    def __init__(self, width=512, height=512):
        self.width, self.height = width, height
        self.image = np.zeros((height, width, 4), dtype=np.uint8)
        self.zbuffer = 1e18 * np.ones((height, width), dtype=float)
        self.coords = np.mgrid[0:height, 0:width].astype(int) 
        
    def clear(self):
        self.image.fill(0)
        self.zbuffer.fill(1e18)

    def triangle(self, t_inv, v0, v1, v2, intensity):
        xmin = int(max(0,               min(v0[0], v1[0], v2[0])))
        xmax = int(min(self.width,       max(v0[0], v1[0], v2[0]) + 1))
        ymin = int(max(0,               min(v0[1], v1[1], v2[1])))
        ymax = int(min(self.height,      max(v0[1], v1[1], v2[1]) + 1))
        
        if xmin >= xmax or ymin >= ymax: return

        P = self.coords[:, ymin:ymax, xmin:xmax].reshape(2, -1)
        P_xy = np.flipud(P)
        
        B = np.dot(t_inv, np.vstack((P_xy, np.ones((1, P_xy.shape[1]), dtype=int))))
        mask = np.all(B >= -1e-4, axis=0) 
        if not np.any(mask): return
        
        X, Y = P_xy[0, mask], P_xy[1, mask]
        Z = v0[2]*B[0, mask] + v1[2]*B[1, mask] + v2[2]*B[2, mask]
        
        valid = Z < self.zbuffer[Y, X]
        if not np.any(valid): return
        
        X_v, Y_v, Z_v = X[valid], Y[valid], Z[valid]
        self.zbuffer[Y_v, X_v] = Z_v
        self.image[Y_v, X_v] = [intensity, intensity, intensity, 255]

    def render_entity(self, entity, view_proj_mat, vp_mat, light_dir_to_src):
        Vh = np.c_[entity.verts, np.ones(len(entity.verts))]
        V_world_h = Vh @ entity.get_model_matrix().T
        V_world = V_world_h[:, :3]

        V_clip = V_world_h @ view_proj_mat.T
        w = V_clip[:, 3]
        valid_v = w > 1e-6
        if not np.any(valid_v): return

        V_ndc = np.zeros((len(V_clip), 3), dtype=float)
        V_ndc[valid_v] = V_clip[valid_v, :3] / w[valid_v, None]

        V_ndc_h = np.c_[V_ndc, np.ones(len(V_ndc))]
        Vs_h = V_ndc_h @ vp_mat.T
        Vs = Vs_h[:, :3]

        faces = entity.faces
        face_valid = np.all(valid_v[faces], axis=1)
        if not np.any(face_valid): return

        Vs_tri = Vs[faces]
        V_tri_world = V_world[faces]

        if len(entity.verts) == 4 and len(entity.faces) == 2:
            # Keep the floor bright enough to stay readable against the dark background.
            intensities = np.full(len(faces), 100, dtype=np.uint8)
        else:
            N = np.cross(V_tri_world[:, 1] - V_tri_world[:, 0], V_tri_world[:, 2] - V_tri_world[:, 0])
            n_mags = np.linalg.norm(N, axis=1, keepdims=True)
            n_mags[n_mags == 0] = 1
            N /= n_mags
            ambient = 60 * entity.color_factor
            diffuse = np.maximum(0, np.dot(N, light_dir_to_src)) * 190 * entity.color_factor
            intensities = np.clip(ambient + diffuse, 0, 255).astype(np.uint8)

        T_mats = np.transpose(Vs_tri, axes=[0, 2, 1]).copy()
        T_mats[:, 2, :] = 1
        try:
            T_mats = np.linalg.inv(T_mats)
        except np.linalg.LinAlgError:
            return

        for i in range(len(faces)):
            if not face_valid[i]: continue
            if intensities[i] <= 0: continue
            
            tri = Vs_tri[i]
            self.triangle(T_mats[i], tri[0], tri[1], tri[2], intensities[i])

def main():
    width, height = 600, 600
    renderer = Renderer(width, height)
    
    bunny_main = Entity("bunny.obj", pos=(0, 0, 0), scale=(1, 1, 1), cull_backface=False)
    bunny_mini = Entity("bunny.obj", pos=(-2.0, 3.0, 0), scale=(0.5, 0.5, 0.5), cull_backface=False)
    
    floor = Entity(
        "floor.obj",
        pos=(0, -1.15, 0),
        rot=(0, 0, 0),
        scale=(5, 5, 5),
        color_factor=1.0,
        cull_backface=False
    )
    
    entities = [bunny_main, bunny_mini, floor] 
    
    light_source_to_dir = np.array([0.5, 3.0, 1.0], dtype=float)
    light_source_to_dir = light_source_to_dir / np.linalg.norm(light_source_to_dir)
    
    eye = np.array([0, 2.2, 6])
    center = np.array([0, -0.8, 0])
    up = np.array([0, 1, 0])

    def lookat(eye, center, up):
        f = (center - eye) / np.linalg.norm(center - eye)
        s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
        u = np.cross(s, f)
        M = np.eye(4)
        M[0, :3], M[1, :3], M[2, :3] = s, u, -f
        M[0, 3], M[1, 3], M[2, 3] = -np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)
        return M

    def perspective(fov, aspect, near, far):
        f = 1.0 / np.tan(np.deg2rad(fov) / 2.0)
        M = np.zeros((4,4))
        M[0,0], M[1,1] = f/aspect, f
        M[2,2], M[2,3] = (far+near)/(near-far), (2*far*near)/(near-far)
        M[3,2] = -1.0
        return M

    def viewport(w, h):
        return np.array([
            [w/2, 0, 0, w/2],
            [0, -h/2, 0, h/2],
            [0, 0, 1000, 1000],
            [0, 0, 0, 1]
        ])

    view_proj_mat = perspective(45, width/height, 0.1, 100.0) @ lookat(eye, center, up)
    vp_mat = viewport(width, height)

    angle = 0
    for i in range(50):
        renderer.clear()
        
        bunny_main.rot[1] = angle 
        
        if bunny_mini.pos[1] > -0.8:
            bunny_mini.pos[1] -= 0.08
            bunny_mini.rot[1] -= 0.1
        else:
            bunny_mini.pos[1] = -0.8
            bunny_mini.rot[1] = 0
            
        angle += 0.05
        
        for ent in entities:
            renderer.render_entity(ent, view_proj_mat, vp_mat, light_source_to_dir)
        
        cv2.imshow("VR Engine - Phase 1 Optimized (600x600)", renderer.image[:, :, :3])
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
