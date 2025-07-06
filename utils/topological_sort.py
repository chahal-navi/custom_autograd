
# topological L-R sort

def build_tree(v: 'Tensor', topo: List):
  """
  inputs a node of the computational graph and outputs a list of the
  subsequent children/dependency nodes, through topological sort.
  """
  if v.prev is not None:
    for children in v.prev:
      build_tree(children, topo)
  topo.append(v)
