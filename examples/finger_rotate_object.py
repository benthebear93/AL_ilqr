from __future__ import annotations

import argparse
from dataclasses import dataclass
import time

import numpy as np

from src.costs import Cost
from src.dynamics import init_dynamics
from src.options import Options
from src.rollout import rollout
from src.solve import solve
from src.solver import (
    get_trajectory,
    initialize_controls,
    initialize_states,
    solver_from_objective,
)


@dataclass(frozen=True)
class FingerRotationConfig:
    horizon: int = 61
    dt: float = 0.05
    finger_origin: tuple[float, float] = (0.0, 0.95)
    link_lengths: tuple[float, float, float] = (0.45, 0.25, 0.26)
    contact_length: float = 0.15
    prismatic_start: float = 0.0
    prismatic_min: float = 0.0
    prismatic_max: float = 0.11
    object_center: tuple[float, float] = (-0.692, 0.913)
    object_goal_center: tuple[float, float] = (-0.642, 0.913)
    object_radius: float = 0.075
    rolling_initial_q: tuple[float, float, float] = (2.6690, 0.2618, 0.3770)
    phi_start: float | None = None
    phi_goal: float | None = None
    elbow_up: bool = True
    state_weights: tuple[float, float, float, float] = (40.0, 30.0, 20.0, 80.0)
    control_weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.25)
    terminal_weights: tuple[float, float, float, float] = (
        400.0,
        300.0,
        250.0,
        600.0,
    )


@dataclass(frozen=True)
class FingerRotationResult:
    config: FingerRotationConfig
    q_ref: np.ndarray
    dx_ref: np.ndarray
    u_ref: np.ndarray
    q_sol: np.ndarray
    dx_sol: np.ndarray
    u_sol: np.ndarray
    phi_ref: np.ndarray
    phi_sol: np.ndarray
    theta_ref: np.ndarray
    theta_sol: np.ndarray
    radial_error: np.ndarray


def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def link_lengths(config: FingerRotationConfig):
    return np.asarray(config.link_lengths, dtype=float)


def finger_origin(config: FingerRotationConfig):
    return np.asarray(config.finger_origin, dtype=float)


def object_center(config: FingerRotationConfig):
    return np.asarray(config.object_center, dtype=float)


def object_goal_center(config: FingerRotationConfig):
    return np.asarray(config.object_goal_center, dtype=float)


def validate_prismatic_position(dx_finger, config: FingerRotationConfig):
    if dx_finger < config.prismatic_min - 1.0e-9:
        raise ValueError("Prismatic displacement is below the minimum bound.")
    if dx_finger > config.prismatic_max + 1.0e-9:
        raise ValueError("Prismatic displacement exceeds the maximum bound.")


def resolved_phi_goal(config: FingerRotationConfig):
    phi_start = resolved_phi_start(config)
    if config.phi_goal is not None:
        return float(config.phi_goal)

    rolling_shift = object_goal_center(config)[0] - object_center(config)[0]
    return phi_start - rolling_shift / config.object_radius


def forward_kinematics(q, config: FingerRotationConfig):
    lengths = link_lengths(config)
    q = np.asarray(q, dtype=float)

    a1 = q[0]
    a2 = q[0] + q[1]
    a3 = a2 + q[2]

    p0 = finger_origin(config)
    p1 = p0 + lengths[0] * np.array([np.cos(a1), np.sin(a1)])
    p2 = p1 + lengths[1] * np.array([np.cos(a2), np.sin(a2)])
    p3 = p2 + lengths[2] * np.array([np.cos(a3), np.sin(a3)])
    return np.vstack([p0, p1, p2, p3]), wrap_to_pi(a3)


def resolved_phi_start(config: FingerRotationConfig):
    if config.phi_start is not None:
        return float(config.phi_start)

    q_init = np.asarray(config.rolling_initial_q, dtype=float)
    return contact_angle(q_init, config.prismatic_start, config)


def prismatic_from_phi(phi, config: FingerRotationConfig):
    dx_finger = config.prismatic_start + config.object_radius * (
        resolved_phi_start(config) - phi
    )
    validate_prismatic_position(dx_finger, config)
    return dx_finger


def object_rotation_from_dx(dx_finger, config: FingerRotationConfig):
    return -(dx_finger - config.prismatic_start) / config.object_radius


def contact_point(q, dx_finger, config: FingerRotationConfig):
    validate_prismatic_position(dx_finger, config)
    joints, heading = forward_kinematics(q, config)
    p2 = joints[2]
    return p2 + (config.contact_length + dx_finger) * np.array(
        [np.cos(heading), np.sin(heading)]
    )


def contact_angle(q, dx_finger, config: FingerRotationConfig):
    rel = contact_point(q, dx_finger, config) - object_center(config)
    return np.arctan2(rel[1], rel[0])


def contact_radial_error(q, dx_finger, config: FingerRotationConfig):
    rel = contact_point(q, dx_finger, config) - object_center(config)
    return np.linalg.norm(rel) - config.object_radius


def desired_contact_pose(phi, config: FingerRotationConfig):
    center = object_center(config)
    contact = center + config.object_radius * np.array([np.cos(phi), np.sin(phi)])
    tip_heading = phi + 0.5 * np.pi
    return contact, wrap_to_pi(tip_heading)


def solve_contact_ik_candidates(phi, dx_finger, config: FingerRotationConfig):
    lengths = link_lengths(config)
    validate_prismatic_position(dx_finger, config)
    contact, tip_heading = desired_contact_pose(phi, config)
    wrist = contact - (config.contact_length + dx_finger) * np.array(
        [np.cos(tip_heading), np.sin(tip_heading)]
    )
    wrist_rel = wrist - finger_origin(config)

    wrist_radius_sq = wrist_rel @ wrist_rel
    l1, l2 = lengths[:2]
    cos_q2 = (wrist_radius_sq - l1**2 - l2**2) / (2.0 * l1 * l2)
    if abs(cos_q2) > 1.0 + 1.0e-9:
        raise ValueError(
            "The requested disk rotation is not reachable with the current finger geometry."
        )
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)

    candidates = []
    for elbow_sign in (1.0, -1.0):
        sin_q2 = elbow_sign * np.sqrt(max(0.0, 1.0 - cos_q2**2))
        q2 = np.arctan2(sin_q2, cos_q2)
        q1 = np.arctan2(wrist_rel[1], wrist_rel[0]) - np.arctan2(
            l2 * sin_q2,
            l1 + l2 * cos_q2,
        )
        q3 = tip_heading - q1 - q2
        candidates.append(np.array([q1, q2, q3], dtype=float))
    return candidates


def closest_angle_branch(q_candidate, q_reference):
    lifted = np.asarray(q_candidate, dtype=float).copy()
    q_reference = np.asarray(q_reference, dtype=float)
    for idx in range(lifted.size):
        delta = lifted[idx] - q_reference[idx]
        lifted[idx] -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
    return lifted


def solve_contact_ik(phi, dx_finger, config: FingerRotationConfig, q_reference=None):
    candidates = solve_contact_ik_candidates(phi, dx_finger, config)
    if q_reference is None:
        q_reference = np.asarray(config.rolling_initial_q, dtype=float)

    lifted_candidates = [
        closest_angle_branch(candidate, q_reference) for candidate in candidates
    ]
    costs = [np.linalg.norm(candidate - q_reference) for candidate in lifted_candidates]
    return lifted_candidates[int(np.argmin(costs))]


def reference_trajectory(config: FingerRotationConfig):
    phi_roll = np.linspace(
        resolved_phi_start(config),
        resolved_phi_goal(config),
        config.horizon,
    )
    dx_roll = np.array([prismatic_from_phi(phi, config) for phi in phi_roll], dtype=float)
    q_roll = []
    q_reference = np.asarray(config.rolling_initial_q, dtype=float)
    for phi, dx_finger in zip(phi_roll, dx_roll):
        q_reference = solve_contact_ik(phi, dx_finger, config, q_reference=q_reference)
        q_roll.append(q_reference.copy())
    q_roll = np.vstack(q_roll)
    x_roll = np.column_stack([q_roll, dx_roll])

    x_ref = x_roll.copy()
    x_start = np.concatenate(
        [np.asarray(config.rolling_initial_q, dtype=float), [config.prismatic_start]]
    )
    bridge_steps = min(max(8, config.horizon // 6), config.horizon - 2)
    for t in range(bridge_steps + 1):
        alpha = t / bridge_steps
        x_ref[t] = (1.0 - alpha) * x_start + alpha * x_roll[t]

    q_ref = x_ref[:, :3]
    dx_ref = x_ref[:, 3]
    phi_ref = np.array(
        [contact_angle(q, dx_finger, config) for q, dx_finger in zip(q_ref, dx_ref)]
    )
    u_ref = np.diff(x_ref, axis=0) / config.dt
    return phi_ref, q_ref, dx_ref, x_ref, u_ref


def build_tracking_costs(config: FingerRotationConfig, x_ref, u_ref):
    q_mat = np.diag(np.asarray(config.state_weights, dtype=float))
    r_mat = np.diag(np.asarray(config.control_weights, dtype=float))
    qf_mat = np.diag(np.asarray(config.terminal_weights, dtype=float))

    costs = []
    for t in range(config.horizon - 1):
        x_ref_t = x_ref[t]
        u_ref_t = u_ref[t]

        def stage_cost(x, u, x_ref_t=x_ref_t, u_ref_t=u_ref_t):
            dx = x - x_ref_t
            du = u - u_ref_t
            return dx @ q_mat @ dx + du @ r_mat @ du

        costs.append(Cost(stage_cost, 4, 4))

    x_ref_terminal = x_ref[-1]

    def terminal_cost(x, u, x_ref_terminal=x_ref_terminal):
        dx = x - x_ref_terminal
        return dx @ qf_mat @ dx

    costs.append(Cost(terminal_cost, 4, 0))
    return costs


def solve_finger_rotation(config: FingerRotationConfig | None = None):
    config = FingerRotationConfig() if config is None else config
    phi_ref, q_ref, dx_ref, x_ref, u_ref = reference_trajectory(config)

    dynamics = [
        init_dynamics(lambda x, u, dt=config.dt: x + dt * u, 4, 4)
        for _ in range(config.horizon - 1)
    ]
    costs = build_tracking_costs(config, x_ref, u_ref)

    x0 = x_ref[0]
    u_bar = np.zeros((config.horizon - 1, 4))
    x_bar = rollout(dynamics[0], x0, u_bar)

    solver = solver_from_objective(
        dynamics,
        costs,
        options=Options(
            verbose=False,
            max_iterations=100,
            lagrangian_gradient_tolerance=1.0e-6,
            objective_tolerance=1.0e-8,
        ),
    )
    initialize_controls(solver, u_bar)
    initialize_states(solver, x_bar)
    solver = solve(solver)

    x_sol, u_sol = get_trajectory(solver)
    x_sol = np.asarray(x_sol, dtype=float)
    u_sol = np.asarray(u_sol, dtype=float)
    q_sol = x_sol[:, :3]
    dx_sol = x_sol[:, 3]
    phi_sol = np.unwrap(
        np.array([contact_angle(q, dx_finger, config) for q, dx_finger in zip(q_sol, dx_sol)])
    )
    theta_ref = np.array(
        [object_rotation_from_dx(dx_finger, config) for dx_finger in dx_ref],
        dtype=float,
    )
    theta_sol = np.array(
        [object_rotation_from_dx(dx_finger, config) for dx_finger in dx_sol],
        dtype=float,
    )
    radial_error = np.array(
        [
            contact_radial_error(q, dx_finger, config)
            for q, dx_finger in zip(q_sol, dx_sol)
        ]
    )

    return FingerRotationResult(
        config=config,
        q_ref=q_ref,
        dx_ref=dx_ref,
        u_ref=u_ref,
        q_sol=q_sol,
        dx_sol=dx_sol,
        u_sol=u_sol,
        phi_ref=phi_ref,
        phi_sol=phi_sol,
        theta_ref=theta_ref,
        theta_sol=theta_sol,
        radial_error=radial_error,
    )


def plot_solution(result: FingerRotationResult):
    import matplotlib.pyplot as plt

    time_state = result.config.dt * np.arange(result.config.horizon)
    time_control = result.config.dt * np.arange(result.config.horizon - 1)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), constrained_layout=True)

    axes[0].plot(time_state, np.rad2deg(result.theta_ref), "--", label="reference")
    axes[0].plot(time_state, np.rad2deg(result.theta_sol), label="optimized")
    axes[0].set_ylabel("Object Rotation [deg]")
    axes[0].set_title("Disk Rotation")
    axes[0].grid(True)
    axes[0].legend()

    for idx in range(3):
        axes[1].plot(
            time_state,
            np.rad2deg(result.q_sol[:, idx]),
            label=f"q{idx + 1}",
        )
    axes[1].set_ylabel("Joint Angle [deg]")
    axes[1].set_title("Finger Joint Trajectory")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(time_state, 1000.0 * result.dx_ref, "--", label="reference")
    axes[2].plot(time_state, 1000.0 * result.dx_sol, label="optimized")
    axes[2].set_ylabel("Prismatic [mm]")
    axes[2].set_title("Sliding Contact Position")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].step(
        time_control,
        result.radial_error[:-1] * 1000.0,
        where="post",
        label="contact radius error",
    )
    for idx in range(4):
        axes[3].step(
            time_control,
            result.u_sol[:, idx],
            where="post",
            label=f"u{idx + 1}",
        )
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Velocity / Error")
    axes[3].set_title("Controls And Contact Error")
    axes[3].grid(True)
    axes[3].legend()

    plt.show()


def segment_transform(p_start, p_end, z_height=0.01):
    delta = p_end - p_start
    angle = np.arctan2(delta[1], delta[0])
    midpoint = 0.5 * (p_start + p_end)
    return np.array([midpoint[0], midpoint[1], z_height]), yaw_to_wxyz(angle)


def yaw_to_wxyz(angle):
    half = 0.5 * angle
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def disk_mesh(radius, height, num_segments=64):
    top_center = np.array([0.0, 0.0, 0.5 * height], dtype=float)
    bottom_center = np.array([0.0, 0.0, -0.5 * height], dtype=float)

    top_ring = []
    bottom_ring = []
    for idx in range(num_segments):
        theta = 2.0 * np.pi * idx / num_segments
        xy = radius * np.array([np.cos(theta), np.sin(theta)], dtype=float)
        top_ring.append(np.array([xy[0], xy[1], 0.5 * height], dtype=float))
        bottom_ring.append(np.array([xy[0], xy[1], -0.5 * height], dtype=float))

    vertices = np.vstack([top_center, bottom_center, *top_ring, *bottom_ring])

    faces = []
    top_offset = 2
    bottom_offset = 2 + num_segments
    for idx in range(num_segments):
        nxt = (idx + 1) % num_segments

        top_i = top_offset + idx
        top_j = top_offset + nxt
        bottom_i = bottom_offset + idx
        bottom_j = bottom_offset + nxt

        faces.append([0, top_i, top_j])
        faces.append([1, bottom_j, bottom_i])
        faces.append([top_i, bottom_i, top_j])
        faces.append([top_j, bottom_i, bottom_j])

    return vertices, np.asarray(faces, dtype=np.int32)


def visualize_solution(result: FingerRotationResult):
    try:
        import viser
    except ImportError as exc:
        raise ImportError(
            "viser is required for visualization. Install project dependencies with `uv sync`."
        ) from exc

    config = result.config
    lengths = link_lengths(config)
    center = object_center(config)

    thickness = 0.012
    object_height = 0.012
    object_size = 2.0 * config.object_radius
    marker_width = 0.004
    marker_height = 0.0015
    marker_z = object_height + 0.5 * marker_height
    object_vertices, object_faces = disk_mesh(config.object_radius, object_height)
    frame_positions = []
    frame_rotations = []
    marker_rotations = []
    contact_positions = []
    for q, dx_finger, theta in zip(result.q_sol, result.dx_sol, result.theta_sol):
        joints, _ = forward_kinematics(q, config)
        frame_positions.append(
            [
                segment_transform(joints[0], joints[1])[0],
                segment_transform(joints[1], joints[2])[0],
                segment_transform(joints[2], joints[3])[0],
            ]
        )
        frame_rotations.append(
            [
                segment_transform(joints[0], joints[1])[1],
                segment_transform(joints[1], joints[2])[1],
                segment_transform(joints[2], joints[3])[1],
            ]
        )
        marker_rotations.append(yaw_to_wxyz(theta))
        contact_xy = contact_point(q, dx_finger, config)
        contact_positions.append(np.array([contact_xy[0], contact_xy[1], marker_z]))

    server = viser.ViserServer()
    server.scene.add_frame("/world", axes_length=0.05, axes_radius=0.0025)
    server.scene.add_mesh_simple(
        "/object/body",
        vertices=object_vertices,
        faces=object_faces,
        color=(64, 125, 199),
        opacity=0.45,
        position=(center[0], center[1], object_height * 0.5),
    )
    object_marker = server.scene.add_box(
        "/object/marker",
        dimensions=(0.92 * object_size, marker_width, marker_height),
        color=(35, 56, 92),
        position=(center[0], center[1], marker_z),
    )
    contact_slider = server.scene.add_box(
        "/finger/contact_slider",
        dimensions=(0.012, 0.012, 0.012),
        color=(179, 43, 43),
        position=tuple(contact_positions[0]),
    )
    link_handles = [
        server.scene.add_box(
            f"/finger/link_{idx + 1}",
            dimensions=(length, thickness, thickness),
            color=color,
        )
        for idx, (length, color) in enumerate(
            zip(
                lengths,
                ((214, 94, 58), (237, 158, 64), (250, 214, 107)),
            )
        )
    ]

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=config.horizon - 1,
            step=1,
            initial_value=0,
        )
        playing = server.gui.add_checkbox("Auto Play", initial_value=True)
        frame_delay = server.gui.add_number(
            "Frame Delay [s]",
            initial_value=config.dt,
            min=0.01,
            step=0.01,
        )

    with server.gui.add_folder("Metrics"):
        rotation_text = server.gui.add_text(
            "Rotation [deg]",
            initial_value=f"{np.rad2deg(result.theta_sol[0]):.2f}",
            disabled=True,
        )
        contact_text = server.gui.add_text(
            "Radius Error [mm]",
            initial_value=f"{1000.0 * result.radial_error[0]:.4f}",
            disabled=True,
        )
        prismatic_text = server.gui.add_text(
            "Prismatic [mm]",
            initial_value=f"{1000.0 * result.dx_sol[0]:.2f}",
            disabled=True,
        )

    def update_scene(frame_idx):
        for handle, position, rotation in zip(
            link_handles,
            frame_positions[frame_idx],
            frame_rotations[frame_idx],
        ):
            handle.position = position
            handle.wxyz = rotation
        contact_slider.position = contact_positions[frame_idx]
        object_marker.wxyz = marker_rotations[frame_idx]
        rotation_text.value = f"{np.rad2deg(result.theta_sol[frame_idx]):.2f}"
        contact_text.value = f"{1000.0 * result.radial_error[frame_idx]:.4f}"
        prismatic_text.value = f"{1000.0 * result.dx_sol[frame_idx]:.2f}"

    @frame_slider.on_update
    def _(_event):
        update_scene(int(frame_slider.value))

    update_scene(0)

    print("Viser server started.")
    print("Open http://localhost:8080 to view the finger and object trajectory.")
    print("Press Ctrl+C to stop the server.")

    try:
        while True:
            if playing.value:
                next_frame = (int(frame_slider.value) + 1) % config.horizon
                frame_slider.value = next_frame
            time.sleep(float(frame_delay.value))
    except KeyboardInterrupt:
        return server


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory optimization for a planar 3-link finger rotating a disk."
    )
    parser.add_argument(
        "--goal-deg",
        type=float,
        default=None,
        help="Clockwise additional rotation magnitude in degrees from the rolling start contact angle. If omitted, use the rolling_trajectory_optimization goal center.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plots.",
    )
    parser.add_argument(
        "--viser",
        action="store_true",
        help="Open Viser visualization.",
    )
    args = parser.parse_args()

    if args.goal_deg is not None:
        base_config = FingerRotationConfig()
        phi_start = resolved_phi_start(base_config)
        config = FingerRotationConfig(phi_goal=phi_start - np.deg2rad(args.goal_deg))
    else:
        config = FingerRotationConfig()
    result = solve_finger_rotation(config)
    target_phi_start = resolved_phi_start(result.config)
    target_phi = resolved_phi_goal(result.config)
    target_rotation = object_rotation_from_dx(
        prismatic_from_phi(target_phi, result.config),
        result.config,
    )
    achieved_rotation = result.theta_sol[-1] - result.theta_sol[0]

    print(f"object start center: {object_center(result.config)}")
    print(f"object goal center: {object_goal_center(result.config)}")
    print(f"contact angle start [deg]: {np.rad2deg(target_phi_start):.2f}")
    print(f"contact angle goal [deg]: {np.rad2deg(target_phi):.2f}")
    print(f"target rotation [deg]: {np.rad2deg(target_rotation):.2f}")
    print(f"achieved rotation [deg]: {np.rad2deg(achieved_rotation):.2f}")
    print(f"prismatic start [mm]: {1000.0 * result.dx_sol[0]:.2f}")
    print(f"prismatic end [mm]: {1000.0 * result.dx_sol[-1]:.2f}")
    print(f"final rotation error [deg]: {np.rad2deg(result.phi_sol[-1] - result.phi_ref[-1]):.4f}")
    print(f"initial contact radius error [mm]: {1000.0 * result.radial_error[0]:.4f}")
    print(f"final contact radius error [mm]: {1000.0 * result.radial_error[-1]:.4f}")

    if not args.no_plot:
        plot_solution(result)
    if args.viser:
        visualize_solution(result)


if __name__ == "__main__":
    main()
