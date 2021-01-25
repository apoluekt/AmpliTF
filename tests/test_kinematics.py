import sys
import tensorflow as tf

sys.path.append("../")

import amplitf.interface as atfi
import amplitf.kinematics as atfk

atfi.set_seed(2)

rndvec = tf.random.uniform([32, 3], dtype=atfi.fptype())

v = rndvec[:, 0]
th = atfi.acos(rndvec[:, 1])
phi = (rndvec[:, 2] * 2 - 1) * atfi.pi()

p = atfk.lorentz_vector(
    atfk.vector(atfi.zeros(v), atfi.zeros(v), atfi.zeros(v)), atfi.ones(v)
)

bp = atfk.lorentz_boost(
    p,
    atfk.rotate_euler(
        atfk.vector(v, atfi.zeros(v), atfi.zeros(v)), th, phi, atfi.zeros(v)
    ),
)

print(bp)
print(atfk.mass(bp))
