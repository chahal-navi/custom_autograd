import numpy as np
from typing import List, NamedTuple, Callable


"""
base Tensor class, input the data as an n dim list
"""

class Tensor():
  def __init__(self, data, label = None, requires_grad = True, prev = None):
    if not isinstance(data, np.ndarray):
      data = np.array(data)

    self.data = data
    self.label = label
    self.requires_grad = requires_grad
    self.prev = prev
    self.shape = self.data.shape
    self.flatten_shape = self.data.flatten().shape[0]
    self.grad_prop = lambda node: None
    self.jacobian = None  # this represents the jacobian of the output wrt the tensor


  def __repr__(self):
    return(f"Tensor object {self.label} with the shape of {self.shape}")


  def __mul__(self, other: 'Tensor'):
    output_value = self.data*other.data
    out = Tensor(output_value, label = self.label + '*' + other.label, prev = [self, other])

    def grad_prop(node):
      if self.jacobian is None:
        self.jacobian = np.zeros((node.flatten_shape, self.flatten_shape))
      if other.jacobian is None:
        other.jacobian = np.zeros((node.flatten_shape, other.flatten_shape))

      self.jacobian += np.matmul(out.jacobian, np.diagflat(other.data.flatten(), k = 0))
      other.jacobian += np.matmul(out.jacobian, np.diagflat(self.data.flatten(), k = 0))

    out.grad_prop = grad_prop
    return out

  def __add__(self, other):
    output_value = self.data + other.data
    out = Tensor(output_value, label = self.label + '+' + other.label, prev = [self, other])

    def grad_prop(node):
      if self.jacobian is None:
        self.jacobian = np.zeros((node.flatten_shape, self.flatten_shape))
      if other.jacobian is None:
        other.jacobian = np.zeros((node.flatten_shape, other.flatten_shape))

      self.jacobian += np.matmul(out.jacobian, np.eye(self.flatten_shape))
      other.jacobian += np.matmul(out.jacobian, np.eye(other.flatten_shape))

    out.grad_prop = grad_prop
    return out

  def __matmul__(self, other):
    output_value = np.matmul(self.data, other.data)
    out = Tensor(output_value, label = self.label + '@' + other.label, prev = [self, other])

    def grad_prop(node):
      if self.jacobian is None:
        self.jacobian = np.zeros((node.flatten_shape, self.flatten_shape))
      if other.jacobian is None:
        other.jacobian = np.zeros((node.flatten_shape, other.flatten_shape))

      print(f"{out.label, out.jacobian.shape} is the jacobian shape")
      print(f"{np.kron( np.transpose(other.data), np.eye(self.shape[0])).shape} is the shape of jacobian before prop of {self.label}")
      print(f"{ np.kron(self.data, np.eye(other.shape[1])).shape} is the shape of jacobian of beofre prop {other.label}")
      self.jacobian += np.matmul(out.jacobian, np.kron(np.eye(self.shape[0]), np.transpose(other.data)))
      other.jacobian += np.matmul(out.jacobian, np.kron(self.data, np.eye(other.shape[1])))


    out.grad_prop = grad_prop

    return out
