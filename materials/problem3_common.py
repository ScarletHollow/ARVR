import cv2
import numpy as np
from core_physics import PhysicsBody, step_simulation
from core_scene import Entity, PhysicsActor, Renderer, lookat, perspective, viewport
from pathlib import Path

def create_problem3_video_scene(engine_dir):
    # Reuse one shared scene for P3.1 and P3.2 so the tracking methods are compared fairly.
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
            "name": "top",
            "position": [0.05, 3.30, 0.00],
            "velocity": [0.00, -0.25, 0.02],
            "radius": 0.56,
            "scale": (0.42, 0.42, 0.42),
            "color": (235, 170, 110),
            "hit_color": (255, 235, 180),
            "spin_rates": (0.8, 1.0, 0.2),
        },
        {
            "name": "left",
            "position": [-2.80, 2.05, -0.45],
            "velocity": [2.45, -0.12, 0.12],
            "radius": 0.58,
            "scale": (0.43, 0.43, 0.43),
            "color": (120, 205, 160),
            "hit_color": (255, 245, 180),
            "spin_rates": (0.4, -1.6, -0.2),
        },
        {
            "name": "right",
            "position": [2.90, 2.15, 0.42],
            "velocity": [-2.35, -0.08, -0.08],
            "radius": 0.58,
            "scale": (0.43, 0.43, 0.43),
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


def run_tracking_video(quaternions, times, title_text, subtitle_text, window_title, frame_step=4):
    engine_dir = Path(__file__).resolve().parent
    width, height = 512, 384
    renderer = Renderer(width, height)
    center_actor, dynamic_actors, floor = create_problem3_video_scene(engine_dir)
    actors = [center_actor] + dynamic_actors
    actor_by_name = {actor.body.name: actor for actor in actors}

    light_dir = np.array([0.6, 2.8, 1.2], dtype=float)
    light_dir /= np.linalg.norm(light_dir)
    eye = np.array([0.0, 2.2, 6.2], dtype=float)
    center = np.array([0.0, -0.35, 0.0], dtype=float)
    up = np.array([0.0, 1.0, 0.0], dtype=float)
    vp_mat = viewport(width, height)
    view_proj_mat = perspective(45.0, width / height, 0.1, 100.0) @ lookat(eye, center, up)

    floor_y = -1.15
    last_collision_text = "No impact yet"
    center_hits = set()
    previous_time = float(times[0])

    for i in range(0, len(quaternions), frame_step):
        current_time = float(times[i])
        dt = max(0.0, current_time - previous_time)
        previous_time = current_time
        elapsed_time = current_time - float(times[0])

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

        center_actor.update_visual(elapsed_time)
        # Only the center bunny follows the IMU orientation; the others are pure physics actors.
        center_actor.entity.rotation_matrix = quaternions[i].to_rotation_matrix()
        renderer.render_entity(center_actor.entity, view_proj_mat, vp_mat, light_dir)

        for actor in dynamic_actors:
            actor.update_visual(elapsed_time)
            renderer.render_entity(actor.entity, view_proj_mat, vp_mat, light_dir)

        renderer.render_entity(floor, view_proj_mat, vp_mat, light_dir)

        cv2.putText(
            renderer.image,
            title_text,
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (230, 230, 230),
            1,
        )
        cv2.putText(
            renderer.image,
            subtitle_text,
            (14, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            renderer.image,
            f"Physics scene | center hits={len(center_hits)}/3 | t={elapsed_time:05.2f}s",
            (14, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (190, 190, 190),
            1,
        )
        cv2.putText(
            renderer.image,
            "Top bunny drops onto the center | side bunnies collide and bounce away",
            (14, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (190, 190, 190),
            1,
        )
        cv2.putText(
            renderer.image,
            f"Last collision: {last_collision_text}",
            (14, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (190, 190, 190),
            1,
        )

        cv2.imshow(window_title, renderer.image[:, :, :3])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

def run_problem3_video(quaternions, times, title_text, subtitle_text, window_title, frame_step=4):
    run_tracking_video(
        quaternions=quaternions,
        times=times,
        title_text=title_text,
        subtitle_text=subtitle_text,
        window_title=window_title,
        frame_step=frame_step,
    )
