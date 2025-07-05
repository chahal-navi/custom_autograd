This is a custom implementation of autograd ( a lighter version than what is used in most deep learning frameworks/libraries ), the data structures are inspired from Andrej Karpathy's implementation and from pytorch's autograd engine.

The engine can produce jacobians of any computational graphs irrespective of the output being scalar or vector

Jacobian Layout:
                If the output is a rank n tensor with m*n*p.......*q dimensions, and the input is another rank l tensor with dimensions as a*b*c*......*f then the jacobian will be an mnp....q times abc...f matrix with i'th column representing the derivative of the flattened output vector wrt the i'th element of the flattened input matrix.
                