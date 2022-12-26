import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numba import jit, njit, prange
from .mesh import Mesh2D
"""
Solver to solve for Laplace/Poisson's equation

"""


def update_boundary(mesh, **kwargs):
    """
    This function updates the boundary conditions of the mesh
    :param mesh: Mesh2D object
    :param kwargs: boundary conditions
    :return: updated mesh
    """
    # default values
    def bcdown(x): return 0.0
    def bcup(x): return 0.0
    def bcleft(x): return 0.0
    def bcright(x): return 0.0
    # update default values with user input
    for key, value in kwargs.items():
        if key == 'bcdown':
            bcdown = value
        elif key == 'bcup':
            bcup = value
        elif key == 'bcleft':
            bcleft = value
        elif key == 'bcright':
            bcright = value
        else:
            raise ValueError("Invalid keyword argument: {}".format(key))

    new_mesh = mesh.mesh.copy()
    # update down boundary
    for i in range(mesh.istartg, mesh.istartg + mesh.buff):
        for j in range(mesh.jstart, mesh.jend+1):
            new_mesh[i, j] = bcdown(mesh.mesh)
    # update up boundary
    for i in range(mesh.iendg - mesh.buff+1, mesh.iendg+1):
        for j in range(mesh.jstart, mesh.jend+1):
            new_mesh[i, j] = bcup(mesh.mesh)
    # update left boundary
    for i in range(mesh.istart, mesh.iend+1):
        for j in range(mesh.jstartg, mesh.jstartg + mesh.buff+1):
            new_mesh[i, j] = bcleft(mesh.mesh)
    # update right boundary
    for i in range(mesh.istart, mesh.iend+1):
        for j in range(mesh.jendg - mesh.buff+1, mesh.jendg+1):
            new_mesh[i, j] = bcright(mesh.mesh)
    return new_mesh


def visualize(mesh):
    plt.clf()
    plt.imshow(mesh, origin='lower')
    plt.pause(0.01)


def relax(mesh: Mesh2D, tolerance, method, visualize_everyloop=False, omega=None, **kwargs):
    if method == 'jacobi':
        method = mesh.jacobi
    elif method == 'gauss_seidel':
        method = mesh.gauss_seidel
    elif method == 'sor':
        method = mesh.sor
    else:
        raise ValueError("Invalid method: {}".format(method))

    error = tolerance + 1
    err = []
    while error > tolerance:
        new_mesh = update_boundary(mesh, **kwargs)
        mesh.mesh, error = method(new_mesh, None if omega is None else omega)
        if visualize_everyloop:
            visualize(mesh.mesh)
        err.append(error)
    visualize(mesh.mesh
    [mesh.istart:mesh.iend, mesh.jstart:mesh.jend]
    )
    return mesh, err


if __name__ == '__main__':
    print("TEST")
