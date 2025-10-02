import jax
import jax.numpy as jnp

from src.dynamics import Dynamics, init_dynamics
from src.rollout import rollout
from src.costs import Cost
from src.constraints import Constraint
from src.solver import (
    get_trajectory,
    solver_from_objective,
    initialize_states,
    initialize_controls,
    solver_from_costs_constraints,
)
from src.solve import solve
from src.options import Options

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from meshcat.animation import Animation
from utils.data_manage import save_outputs

# horizon
T = 51
num_state = 3
num_action = 2


# continuous and discrete dynamics
def car_continuous(x, u):
    return jnp.array([u[0] * jnp.cos(x[2]), u[0] * jnp.sin(x[2]), u[1]])


def car_discrete(x, u):
    h = 0.1
    return x + h * car_continuous(x + 0.5 * h * car_continuous(x, u), u)


# model
car = init_dynamics(car_discrete, num_state, num_action)
dynamics = [car for _ in range(T - 1)]

# initial and goal states
x1 = jnp.array([0.0, 0.0, 0.0])
xT = jnp.array([1.0, 1.0, 0.0])

# rollout with small controls
u_bar = [0.01 * jnp.array([1.0, 0.1]) for _ in range(T - 1)]
x_bar = rollout(car, x1, jnp.stack(u_bar))

# objective
objective = [
    Cost(
        lambda x, u: jnp.dot(x - xT, x - xT) + 0.01 * jnp.dot(u, u),
        num_state,
        num_action,
    )
    for _ in range(T - 1)
]
objective.append(Cost(lambda x, u: 1000.0 * jnp.dot(x - xT, x - xT), num_state, 0))
# constraints
ul = -5.0 * jnp.ones(num_action)
uu = 5.0 * jnp.ones(num_action)

p_obs = jnp.array([0.5, 0.5])
r_obs = 0.1

constraints = []
for _ in range(T - 1):

    def constraint_stage(x, u):
        e = x[:2] - p_obs
        return jnp.concatenate(
            [
                ul - u,  # lower bound
                u - uu,  # upper bound
                jnp.array([r_obs**2 - jnp.dot(e, e)]),  # obstacle
            ]
        )

    constraints.append(
        Constraint(
            constraint_stage, num_state, num_action, indices_inequality=list(range(5))
        )
    )


def constraint_terminal(x, u):
    e = x[:2] - p_obs
    return jnp.concatenate(
        [x - xT, jnp.array([r_obs**2 - jnp.dot(e, e)])]  # goal  # obstacle
    )


constraints.append(
    Constraint(
        constraint_terminal, num_state, num_action, indices_inequality=[3]
    )  # obstacle only inequality
)

# solver
options = Options(verbose=True, max_iterations=100)
solver = solver_from_costs_constraints(
    dynamics, objective, constraints=constraints, options=options
)

initialize_controls(solver, jnp.stack(u_bar))
initialize_states(solver, jnp.stack(x_bar))

# solve
solve(solver)

# extract trajectory
x_sol, u_sol = get_trajectory(solver)

import matplotlib.pyplot as plt
import numpy as np

x_sol_np = np.array(x_sol)
u_sol_np = np.array(u_sol)

save_outputs(x_sol=x_sol_np, u_sol=u_sol_np)

x_sol_np = np.array(x_sol)  # shape: (T, num_state)
plt.figure(figsize=(8, 4))
for i in range(x_sol_np.shape[1]):
    plt.plot(x_sol_np[:, i], label=f"x[{i}]")
plt.xlabel("Time step")
plt.ylabel("State values")
plt.title("State Trajectory")
plt.legend()
plt.grid(True)

u_sol_np = np.array(u_sol)  # shape: (T-1, num_action)
plt.figure(figsize=(8, 4))
for i in range(u_sol_np.shape[1]):
    plt.step(range(u_sol_np.shape[0]), u_sol_np[:, i], where="post", label=f"u[{i}]")
plt.xlabel("Time step")
plt.ylabel("Control values")
plt.title("Control Trajectory")
plt.legend()
plt.grid(True)

plt.show()


def car_scene_objects(goal, obs, obs_r):
    return {
        "car": {
            "geometry": g.Box([0.2, 0.1, 0.05]),
            "color": (0.2, 0.2, 1.0, 1.0),
        },
        "goal": {
            "geometry": g.Sphere(0.05),
            "color": (0.0, 1.0, 0.0, 1.0),
            "transform": tf.translation_matrix([goal[0], goal[1], 0.0]),
        },
        "obstacle": {
            "geometry": g.Sphere(obs_r),
            "color": (1.0, 0.0, 0.0, 0.5),
            "transform": tf.translation_matrix([obs[0], obs[1], 0.0]),
        },
    }


def trajectory_to_transforms(x_sol):
    """
    (T,3) trajectory -> list of 4x4 transforms
    """
    traj = []
    for x, y, theta in x_sol:
        T = tf.translation_matrix([x, y, 0.025]) @ tf.rotation_matrix(  # 살짝 띄우기
            theta, [0, 0, 1]
        )
        traj.append(T)
    return {"car": traj}


def visualize_scene(vis, objects, trajectories, dt=0.1):
    for name, cfg in objects.items():
        geom = cfg["geometry"]
        color = cfg.get("color", (0.5, 0.5, 0.5, 1.0))
        transform = cfg.get("transform", np.eye(4))

        vis[name].set_object(
            geom,
            g.MeshPhongMaterial(
                color=_rgba_to_hex(color),
                transparency=1.0 - color[3],
            ),
        )
        vis[name].set_transform(transform)

    anim = Animation()
    anim.default_framerate = int(1.0 / dt)
    T = max(len(traj) for traj in trajectories.values())

    for t in range(T):
        for name, traj in trajectories.items():
            if t < len(traj):
                with anim.at_frame(vis[name], t) as f:
                    f.set_transform(traj[t])

    vis.set_animation(anim)


def _rgba_to_hex(rgba):
    rgb255 = tuple(int(255 * c) for c in rgba[:3])
    return (rgb255[0] << 16) + (rgb255[1] << 8) + rgb255[2]


vis = meshcat.Visualizer().open()

objects = car_scene_objects(xT, p_obs, r_obs)

trajectories = trajectory_to_transforms(np.array(x_sol))

visualize_scene(vis, objects, trajectories, dt=0.1)
