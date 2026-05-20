from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


AIR_DENSITY = 1.3
DRAG_COEFFICIENT = 0.5
CROSS_SECTION_AREA = 0.2
DEFAULT_GRAVITY = np.array([0.0, -9.81, 0.0], dtype=float)
EPSILON = 1e-8


@dataclass
class PhysicsBody:
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    radius: float
    name: str = ""
    is_static: bool = False
    drag_enabled: bool = True
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)
        self.acceleration = np.asarray(self.acceleration, dtype=float).reshape(3)
        self.mass = float(self.mass)
        self.radius = float(self.radius)
        if self.mass <= 0.0:
            raise ValueError("mass must be positive.")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive.")

    def copy(self):
        return PhysicsBody(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass,
            radius=self.radius,
            name=self.name,
            is_static=self.is_static,
            drag_enabled=self.drag_enabled,
            acceleration=self.acceleration.copy(),
        )


def drag_force(velocity, air_density=AIR_DENSITY, drag_coefficient=DRAG_COEFFICIENT, area=CROSS_SECTION_AREA):
    velocity = np.asarray(velocity, dtype=float).reshape(3)
    speed = float(np.linalg.norm(velocity))
    if speed <= EPSILON:
        return np.zeros(3, dtype=float)

    magnitude = 0.5 * float(air_density) * (speed ** 2) * float(drag_coefficient) * float(area)
    return -(velocity / speed) * magnitude


def compute_acceleration(body, gravity=DEFAULT_GRAVITY):
    gravity = np.asarray(gravity, dtype=float).reshape(3)
    if body.is_static:
        body.acceleration = np.zeros(3, dtype=float)
        return body.acceleration.copy()

    acceleration = gravity.copy()
    if body.drag_enabled:
        acceleration += drag_force(body.velocity) / body.mass

    body.acceleration = acceleration
    return acceleration.copy()


def integrate_body(body, dt, gravity=DEFAULT_GRAVITY):
    dt = float(dt)
    if dt < 0.0:
        raise ValueError("dt must be non-negative.")

    if body.is_static or dt == 0.0:
        body.acceleration = np.zeros(3, dtype=float)
        return body.acceleration.copy()

    acceleration = compute_acceleration(body, gravity=gravity)
    body.velocity = body.velocity + acceleration * dt
    body.position = body.position + body.velocity * dt
    return acceleration.copy()


def detect_sphere_collision(body_a, body_b):
    delta = body_b.position - body_a.position
    distance = float(np.linalg.norm(delta))
    min_distance = float(body_a.radius + body_b.radius)
    if distance <= EPSILON:
        normal = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        normal = delta / distance

    return {
        "collided": distance < min_distance,
        "distance": distance,
        "min_distance": min_distance,
        "normal": normal,
        "penetration": max(0.0, min_distance - distance),
    }


def reflect_velocity_along_normal(velocity, normal):
    velocity = np.asarray(velocity, dtype=float).reshape(3)
    normal = np.asarray(normal, dtype=float).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm <= EPSILON:
        return velocity.copy()
    normal = normal / norm
    return velocity - 2.0 * np.dot(velocity, normal) * normal


def resolve_sphere_collision(body_a, body_b):
    collision = detect_sphere_collision(body_a, body_b)
    if not collision["collided"]:
        return False, collision

    normal = collision["normal"]
    penetration = collision["penetration"]

    inverse_mass_a = 0.0 if body_a.is_static else 1.0 / body_a.mass
    inverse_mass_b = 0.0 if body_b.is_static else 1.0 / body_b.mass
    inverse_mass_sum = inverse_mass_a + inverse_mass_b

    if inverse_mass_sum > EPSILON and penetration > 0.0:
        correction = normal * penetration
        if inverse_mass_a > 0.0:
            body_a.position = body_a.position - correction * (inverse_mass_a / inverse_mass_sum)
        if inverse_mass_b > 0.0:
            body_b.position = body_b.position + correction * (inverse_mass_b / inverse_mass_sum)

    relative_velocity = body_a.velocity - body_b.velocity
    closing_speed = float(np.dot(relative_velocity, normal))
    if closing_speed > 0.0:
        if not body_a.is_static and float(np.dot(body_a.velocity, normal)) > 0.0:
            body_a.velocity = reflect_velocity_along_normal(body_a.velocity, normal)
        if not body_b.is_static and float(np.dot(body_b.velocity, normal)) < 0.0:
            body_b.velocity = reflect_velocity_along_normal(body_b.velocity, normal)

    collision["closing_speed"] = closing_speed
    collision["names"] = (body_a.name, body_b.name)
    return True, collision


def resolve_floor_collision(body, floor_y, restitution=0.35, tangential_damping=0.96):
    if body.is_static:
        return False

    floor_contact_y = float(floor_y) + body.radius
    if body.position[1] >= floor_contact_y:
        return False

    body.position[1] = floor_contact_y
    if body.velocity[1] < 0.0:
        body.velocity[1] = -body.velocity[1] * float(restitution)
    body.velocity[0] *= float(tangential_damping)
    body.velocity[2] *= float(tangential_damping)
    return True


def step_simulation(
    bodies,
    dt,
    gravity=DEFAULT_GRAVITY,
    floor_y=None,
    floor_restitution=0.35,
    floor_tangential_damping=0.96,
):
    for body in bodies:
        integrate_body(body, dt, gravity=gravity)

    collisions = []
    for index_a, index_b in combinations(range(len(bodies)), 2):
        collided, info = resolve_sphere_collision(bodies[index_a], bodies[index_b])
        if collided:
            collisions.append(info)

    floor_hits = []
    if floor_y is not None:
        for body in bodies:
            if resolve_floor_collision(
                body,
                floor_y=floor_y,
                restitution=floor_restitution,
                tangential_damping=floor_tangential_damping,
            ):
                floor_hits.append(body.name)

    return {
        "collisions": collisions,
        "floor_hits": floor_hits,
    }
