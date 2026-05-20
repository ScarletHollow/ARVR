from pathlib import Path

import cv2
import numpy as np

from core_physics import PhysicsBody, step_simulation
from core_scene import Entity, PhysicsActor, Renderer, lookat, perspective, viewport


def save_png(path, image_bgr):
    ok, encoded = cv2.imencode(".png", image_bgr)
    if ok:
        encoded.tofile(path)
    return ok


def create_demo_scene(engine_dir):
    bunny_path = engine_dir / "bunny.obj"

    center_actor = PhysicsActor(
        entity=Entity(bunny_path, pos=(0.0, 0.0, 0.0), scale=(0.95, 0.95, 0.95), color=(220, 215, 205)),
        body=PhysicsBody(
            position=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            mass=5.0,
            radius=1.05,
            name="center",
            is_static=True,
            drag_enabled=False,
        ),
        spin_rates=(0.0, 0.0, 0.0),
        hit_color=(255, 245, 180),
    )

    dynamic_specs = [
        {
            "name": "left",
            "position": [-2.4, 2.7, -0.8],
            "velocity": [2.05, -0.15, 0.18],
            "radius": 0.60,
            "scale": (0.45, 0.45, 0.45),
            "color": (235, 130, 100),
            "hit_color": (255, 235, 180),
            "spin_rates": (1.0, 1.8, 0.3),
        },
        {
            "name": "right",
            "position": [2.55, 3.05, 0.55],
            "velocity": [-2.00, -0.10, -0.08],
            "radius": 0.58,
            "scale": (0.43, 0.43, 0.43),
            "color": (120, 205, 160),
            "hit_color": (255, 245, 180),
            "spin_rates": (0.4, -1.6, -0.2),
        },
        {
            "name": "rear",
            "position": [-0.65, 3.35, 1.85],
            "velocity": [0.22, -0.18, -1.70],
            "radius": 0.56,
            "scale": (0.41, 0.41, 0.41),
            "color": (115, 155, 240),
            "hit_color": (255, 245, 180),
            "spin_rates": (-0.6, 1.2, 1.1),
        },
    ]

    dynamic_actors = []
    for spec in dynamic_specs:
        dynamic_actors.append(
            PhysicsActor(
                entity=Entity(bunny_path, pos=spec["position"], scale=spec["scale"], color=spec["color"]),
                body=PhysicsBody(
                    position=spec["position"],
                    velocity=spec["velocity"],
                    mass=1.0,
                    radius=spec["radius"],
                    name=spec["name"],
                ),
                spin_rates=spec["spin_rates"],
                hit_color=spec["hit_color"],
            )
        )

    floor = Entity(engine_dir / "floor.obj", pos=(0.0, -1.15, 0.0), scale=(5.0, 1.0, 5.0), color=(150, 150, 150))
    return center_actor, dynamic_actors, floor


def run_problem5_demo(engine_dir=None):
    if engine_dir is None:
        engine_dir = Path(__file__).resolve().parent
    else:
        engine_dir = Path(engine_dir)

    preview_path = engine_dir / "phase5_physics_preview.png"
    final_frame_path = engine_dir / "phase5_physics_final.png"

    width, height = 512, 384
    renderer = Renderer(width, height)
    center_actor, dynamic_actors, floor = create_demo_scene(engine_dir)
    actors = [center_actor] + dynamic_actors
    actor_by_name = {actor.body.name: actor for actor in actors}

    light_dir = np.array([0.6, 2.8, 1.2], dtype=float)
    light_dir /= np.linalg.norm(light_dir)
    eye = np.array([0.0, 2.2, 6.2], dtype=float)
    center = np.array([0.0, -0.35, 0.0], dtype=float)
    up = np.array([0.0, 1.0, 0.0], dtype=float)
    vp_mat = viewport(width, height)
    view_proj_mat = perspective(45.0, width / height, 0.1, 100.0) @ lookat(eye, center, up)

    dt = 1.0 / 60.0
    total_frames = 96
    floor_y = -1.15
    gui_enabled = True
    preview_saved = False
    last_collision_text = "No impact yet"
    center_hits = set()

    for frame in range(total_frames):
        elapsed_time = frame * dt
        step_info = step_simulation(
            [actor.body for actor in actors],
            dt=dt,
            floor_y=floor_y,
            floor_restitution=0.32,
            floor_tangential_damping=0.97,
        )

        if step_info["collisions"]:
            labels = []
            for collision in step_info["collisions"]:
                body_a, body_b = collision["names"]
                labels.append(f"{body_a}/{body_b}")
                actor_by_name[body_a].notify_collision()
                actor_by_name[body_b].notify_collision()
                if "center" in (body_a, body_b):
                    center_hits.add(body_b if body_a == "center" else body_a)
            last_collision_text = ", ".join(labels[:2])

        renderer.clear()
        for actor in actors:
            actor.update_visual(elapsed_time)
            renderer.render_entity(actor.entity, view_proj_mat, vp_mat, light_dir)
        renderer.render_entity(floor, view_proj_mat, vp_mat, light_dir)

        cv2.putText(
            renderer.image,
            "Problem 5: Gravity + Drag + Sphere Collision",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (230, 230, 230),
            2,
        )
        cv2.putText(
            renderer.image,
            f"Center bunny fixed | center hits={len(center_hits)}/3 | frame={frame + 1}/{total_frames}",
            (16, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            renderer.image,
            "Falling bunnies use manual gravity, drag, velocity, and position updates",
            (16, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (190, 190, 190),
            1,
        )
        cv2.putText(
            renderer.image,
            f"Last collision: {last_collision_text}",
            (16, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (190, 190, 190),
            1,
        )

        if not preview_saved and len(center_hits) == 3:
            # Save the report/README preview once the intended three center collisions have all happened.
            preview_saved = save_png(str(preview_path), renderer.image[:, :, :3])

        if gui_enabled:
            try:
                cv2.imshow("VR - Problem 5: Physics Demo", renderer.image[:, :, :3])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                gui_enabled = False
                print("OpenCV GUI unavailable. Continuing headless and saving preview frames.")

    save_png(str(final_frame_path), renderer.image[:, :, :3])
    if gui_enabled:
        cv2.destroyAllWindows()

    if not preview_saved:
        save_png(str(preview_path), renderer.image[:, :, :3])

    final_speed_summary = {
        actor.body.name: float(np.linalg.norm(actor.body.velocity))
        for actor in dynamic_actors
    }
    print(
        "Problem 5 summary | "
        "floor_restitution=0.32 | floor_tangential_damping=0.97 | "
        f"center_hits={len(center_hits)}/3 | "
        f"final_dynamic_speed_magnitudes={final_speed_summary}"
    )


def main():
    run_problem5_demo()


if __name__ == "__main__":
    main()
