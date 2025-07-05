{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8c48b73f-21e6-4bf8-9e9f-38c58c637953",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "\n",
    "# topological L-R sort\n",
    "\n",
    "def build_tree(v: 'Tensor', topo: List):\n",
    "  \"\"\"\n",
    "  inputs a node of the computational graph and outputs a list of the\n",
    "  subsequent children/dependency nodes, through topological sort.\n",
    "  \"\"\"\n",
    "  if v.prev is not None:\n",
    "    for children in v.prev:\n",
    "      build_tree(children, topo)\n",
    "  topo.append(v)\n"
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
