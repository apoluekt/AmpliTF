import uproot

def write_tuple(rootfile, array, branches, tree="tree") : 
  with uproot.recreate(rootfile, compression=uproot.ZLIB(4)) as file :  
    file[tree] = uproot.newtree( { b : "float64" for b in branches } )
    d = { b : array[:,i] for i,b in enumerate(branches) }
    print(d)
    file[tree].extend(d)
