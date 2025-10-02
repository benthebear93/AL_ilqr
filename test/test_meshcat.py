import meshcat.geometry as g
import meshcat.transformations as tf

# Meshcat visualizer
import meshcat
from ..visulization.meshcat_object import visualize
vis = meshcat.Visualizer().open()

objects = {
    "box": {
        "geometry": g.Box([0.2, 0.2, 0.2]),
        "color": (0, 0, 1, 1),
    },
    "pusher": {
        "geometry": g.Sphere(0.05),
        "color": (1, 0, 0, 1),
    },
}

# Trajectories
T = 50
box_traj = []
pusher_traj = []
for t in range(T):
    box_traj.append(tf.translation_matrix([0.01 * t, 0, 0]))
    pusher_traj.append(tf.translation_matrix([0, 0.01 * t, 0.05]))

trajectories = {
    "box": box_traj,
    "pusher": pusher_traj,
}

visualize(vis, objects, trajectories, dt=0.1)
