import sys
sys.path.append("../")

import amplitf.interface as atfi

atfi.backend_numpy()

import amplitf.kinematics as atfk
import numpy as np

p1 = atfi.const( [ [0.514208, -0.184219, -0.184219, 1.35527], [0.514208, -0.184219, -0.184219, 1.35527] ] )
p2 = atfi.const( [ [-0.305812, 0.284, 0.284, 0.755744], [-0.305812, 0.284, 0.284, 0.755744] ])
p3 = atfi.const( [ [-0.061663, -0.0211864, -0.0211864, 0.208274], [-0.061663, -0.0211864, -0.0211864, 0.208274] ] )
p4 = atfi.const( [ [-0.146733, -0.0785946, -0.0785946, 0.777613], [-0.146733, -0.0785946, -0.0785946, 0.777613] ] )

print(p1.shape)

angles = atfk.nested_helicity_angles([[[p1, p2], p3], p4])
print(angles)

vectors = [
    atfi.const([0.514208, -0.184219, -0.184219, 1.35527]),
    atfi.const([-0.305812, 0.284, 0.284, 0.755744]),
    atfi.const([-0.061663, -0.0211864, -0.0211864, 0.208274]),
    atfi.const([-0.146733, -0.0785946, -0.0785946, 0.777613])
]

print(np.sum(vectors, axis=0))

angles = atfk.nested_helicity_angles(
    [[[ vectors[2], vectors[3] ], vectors[1] ], vectors[0] ])

print(np.cos(angles[0]), angles[1], np.cos(angles[2]), angles[3], np.cos(angles[4]), angles[5])
