class backprop():
  """
  this object will call a tarverser function defined in itself
  that fucntion will store all the jacobians of all the intermediate values
  which have requires_grad as True
  """
  def __init__(self, v: 'Tensor'):
    self.node = v
    self.topological_tree = []
    build_tree(self.node, self.topological_tree)
    self.topological_tree.reverse()
    print(self.topological_tree)
    def propogator(self):
      self.node.jacobian = np.eye(self.node.flatten_shape)
      for i in self.topological_tree:
        i.grad_prop(self.node)
    propogator(self)

