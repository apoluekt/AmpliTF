import sys
import tensorflow as tf

sys.path.append("../")

import amplitf.kinematics as atf_kin
import numpy as np

p1 = np.array(
    [
        [0.514208, -0.184219, -0.184219, 1.35527],
        [0.514208, -0.184219, -0.184219, 1.35527],
    ]
)
p2 = np.array(
    [[-0.305812, 0.284, 0.284, 0.755744], [-0.305812, 0.284, 0.284, 0.755744]]
)
p3 = np.array(
    [
        [-0.061663, -0.0211864, -0.0211864, 0.208274],
        [-0.061663, -0.0211864, -0.0211864, 0.208274],
    ]
)
p4 = np.array(
    [
        [-0.146733, -0.0785946, -0.0785946, 0.777613],
        [-0.146733, -0.0785946, -0.0785946, 0.777613],
    ]
)

print(p1.shape)

angles = atf_kin.nested_helicity_angles([[[p1, p2], p3], p4])
print(angles)

vectors = [
    (0.514208, -0.184219, -0.184219, 1.35527),
    (-0.305812, 0.284, 0.284, 0.755744),
    (-0.061663, -0.0211864, -0.0211864, 0.208274),
    (-0.146733, -0.0785946, -0.0785946, 0.777613),
]
print(np.sum(vectors, axis=0))
angles = atf_kin.nested_helicity_angles(
    [
        [
            [tf.convert_to_tensor(vectors[2]), tf.convert_to_tensor(vectors[3])],
            tf.convert_to_tensor(vectors[1]),
        ],
        tf.convert_to_tensor(vectors[0]),
    ]
)

print(
    np.cos(angles[0].numpy()),
    angles[1].numpy(),
    np.cos(angles[2].numpy()),
    angles[3].numpy(),
    np.cos(angles[4].numpy()),
    angles[5].numpy(),
)
