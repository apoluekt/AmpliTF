import sys

def backend_numpy() : 
  from amplitf.interface import interface_numpy
  m = sys.modules[__name__]
  for k in dir(interface_numpy) : 
    if str(k)[:2] == "__" : continue
    m.__dict__[k] = interface_numpy.__dict__[k]
  print("Imported NumPy backend. ")

def backend_jax() : 
  from amplitf.interface import interface_jax
  m = sys.modules[__name__]
  for k in dir(interface_jax) : 
    if str(k)[:2] == "__" : continue
    m.__dict__[k] = interface_jax.__dict__[k]
  print("Imported JAX backend. ")

def backend_tf() : 
  from amplitf.interface import interface_tf
  m = sys.modules[__name__]
  for k in dir(interface_tf) : 
    if str(k)[:2] == "__" : continue
    m.__dict__[k] = interface_tf.__dict__[k]
  print("Imported TensorFlow backend. ")

def backend_auto() : 
  import importlib
  tf_spec = importlib.util.find_spec("tensorflow")
  if tf_spec : 
    backend_tf()
    return 
  jax_spec = importlib.util.find_spec("jax")
  if jax_spec : 
    backend_jax()
    return
  backend_numpy()

