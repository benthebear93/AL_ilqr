import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation
from utils.data_manage import load_x_sol_from
from pathlib import Path

import time


def trajectory_to_transforms(x_sol):
    """
    (T,3) trajectory -> list of 4x4 transforms
    """
    traj = []
    for x, y, theta in x_sol:
        T = tf.translation_matrix([x, y, 0.025]) @ tf.rotation_matrix(theta, [0, 0, 1])
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
        time.sleep(1)

    vis["/Background"].set_property("visible", True)
    vis["/Background"].set_property("top_color", [0.98, 0.98, 0.98])
    vis["/Background"].set_property("bottom_color", [0.98, 0.98, 0.98])
    cam_tf = (
        tf.translation_matrix([0.0, 0.0, 75.0])
        @ tf.rotation_matrix(0.5 * np.pi, [0, 0, 1])
        @ tf.rotation_matrix(-np.pi / 2.5, [0, 1, 0])
    )
    # vis["/Cameras/default"].set_transform(cam_tf)

    # vis["/Cameras/default/rotated/<object>"].set_property("zoom", 40)

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


if __name__ == "__main__":
    here = Path(__file__).resolve().parent  # .../examples
    data_dir = (here.parent / "data").resolve()  # .../data

    x_sol = load_x_sol_from(data_dir)

    vis = meshcat.Visualizer().open()

    objects = {
        "car": {
            "geometry": g.Box([0.2, 0.1, 0.05]),
            "color": (0.2, 0.2, 1.0, 1.0),
        }
    }

    trajectories = trajectory_to_transforms(x_sol)
    visualize_scene(vis, objects, trajectories, dt=0.1)
