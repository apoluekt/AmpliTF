import sys

sys.path.append("../")
import amplitf.interface as atfi

atfi.backend_auto()

res = atfi.max([1,1], [2,0])
print(res)

atfi.backend_numpy()

res = atfi.max([1,1], [2,0])
print(res)

