import time
import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation


def visualize(
    vis,
    objects,
    trajectories,
    dt=0.1,
    background_color=(0.98, 0.98, 0.98),
    camera_distance=50.0,
    zoom=50,
):
    # --- Create objects ---
    for name, cfg in objects.items():
        geometry = cfg["geometry"]
        color = cfg.get("color", (0.5, 0.5, 0.5, 1.0))
        transform = cfg.get("transform", np.eye(4))

        vis[name].set_object(
            geometry,
            g.MeshPhongMaterial(color=_rgba_to_hex(color), transparency=1.0 - color[3]),
        )
        vis[name].set_transform(transform)

    time.sleep(1)

    # --- Background / camera ---
    vis["/Background"].set_property("visible", True)
    vis["/Background"].set_property("top_color", background_color)
    vis["/Background"].set_property("bottom_color", background_color)

    cam_tf = (
        tf.translation_matrix([0.0, 0.0, camera_distance])
        @ tf.rotation_matrix(0.5 * np.pi, [0, 0, 1])
        @ tf.rotation_matrix(-np.pi / 2.5, [0, 1, 0])
    )
    vis["/Cameras/default"].set_transform(cam_tf)
    vis["/Cameras/default/rotated/<object>"].set_property("zoom", zoom)

    # --- Animation ---
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
