import sys
import tensorflow as tf

sys.path.append("../")

import amplitf.interface as atfi

print( atfi.__dir__() )

res = atfi.max(1, 2)
print(res)
