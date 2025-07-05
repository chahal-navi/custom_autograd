{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef62160f-d715-4399-a4c4-3380bef65cb1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "class backprop():\n",
    "  \"\"\"\n",
    "  this object will call a tarverser function defined in itself\n",
    "  that fucntion will store all the jacobians of all the intermediate values\n",
    "  which have requires_grad as True\n",
    "  \"\"\"\n",
    "  def __init__(self, v: 'Tensor'):\n",
    "    self.node = v\n",
    "    self.topological_tree = []\n",
    "    build_tree(self.node, self.topological_tree)\n",
    "    self.topological_tree.reverse()\n",
    "    print(self.topological_tree)\n",
    "    def propogator(self):\n",
    "      self.node.jacobian = np.eye(self.node.flatten_shape)\n",
    "      for i in self.topological_tree:\n",
    "        i.grad_prop(self.node)\n",
    "    propogator(self)\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
