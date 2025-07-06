
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

"""
block diagonal matrix fucntion
used to create the jacobian for matrix multiplication.
for flattened marices
"""

def block_diag_matrix(matrix, shape):
  block_diag_matrix = np.zeros(shape)
  block_num = shape[0]/matrix.shape[0]
  print(block_num, 'is the block number')
  i = 0
  while i < block_num:
    block_diag_matrix[i*matrix.shape[0] : matrix.shape[0]*(i + 1), i*matrix.shape[1] : matrix.shape[1] * (i + 1)] = matrix
    i += 1
  return block_diag_matrix

