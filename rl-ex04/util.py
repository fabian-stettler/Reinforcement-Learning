# some utilities from previous exercises for plotting etc.

import numpy as np
import math
import matplotlib.pyplot as plt
from mdp import RState, MDPGridworld

def policy_to_direction(p: np.array) -> (np.array, np.array):
    """
    Calculate the direction of the max value of the policy  into two arrays that can be plotted.
    """
    u = np.zeros((p.shape[0]), dtype=float)
    v = np.zeros((p.shape[0]), dtype=float)
    for i in range(p.shape[0]):
        direction = np.argmax(p[i,:])
        if p[i, direction] > 0:
            if direction == MDPGridworld.N:
                u[i] = 0.0
                v[i] = 1.0
            if direction == MDPGridworld.E:
                u[i] = 1.0
                v[i] = 0.0
            if direction == MDPGridworld.S:
                u[i] = 0.0
                v[i] = -1.0
            if direction == MDPGridworld.W:
                u[i] = -1.0
                v[i] = 0.0
    return u,v

def plot_policy_values(p, values, g):
    """
    Plot the policy as arrows, and the values as text using the render function from the gridworld g
    """
    fig, ax = plt.subplots()
    g.render(values, ax)

    u,v = policy_to_direction(p)
    u = u.reshape((g.height, g.width))
    v = v.reshape((g.height, g.width))
    ax.quiver(u, v)

    
def max_arg_with_ties(q):
    max_value = -math.inf
    tied_index = []
    
    for i in range(len(q)):
        if q[i] > max_value:
            max_value = q[i]
            tied_index = [i]
        elif q[i] == max_value:
            tied_index.append(i)
    return np.random.choice(tied_index)
    