"""
This file define classes for generating 2D meshes.

"""
import numpy as np
from numba import jit, int32, float64, njit, prange, get_num_threads, set_num_threads
from numba.experimental import jitclass

num_of_threads = get_num_threads()//2
set_num_threads(num_of_threads)

class Mesh2D:
    def __init__(self, nx, ny, buff, **kwargs):
        """
        This function generates a 2D mesh with nx*ny nodes and buff layers of ghost nodes.
        The mesh is stored in a dictionary with the following keys:
        - xmin: minimum x coordinate
        - xmax: maximum x coordinate
        - ymin: minimum y coordinate
        - ymax: maximum y coordinate
        - nx: number of nodes in x direction
        - ny: number of nodes in y direction
        - buff: number of ghost nodes
        - nxgc: number of nodes in x direction including ghost nodes
        - nygc: number of nodes in y direction including ghost nodes
        - dx: grid spacing in x direction
        - dy: grid spacing in y direction
        - istart: index of the first node in x direction
        - iend: index of the last node in x direction
        - jstart: index of the first node in y direction
        - jend: index of the last node in y direction
        - istartg: index of the first ghost node in x direction
        - iendg: index of the last ghost node in x direction
        - jstartg: index of the first ghost node in y direction
        - jendg: index of the last ghost node in y direction

        :param nx: number of nodes in x direction
        :param ny: number of nodes in y direction
        :param buff: number of ghost nodes
        :param kwargs: xmin, xmax, ymin, ymax
        """
        # default values
        self._xmin = 0.0
        self._xmax = 1.0
        self._ymin = 0.0
        self._ymax = 1.0
        # update default values with user input
        for key, value in kwargs.items():
            if key == 'xmin':
                self._xmin = value
            elif key == 'xmax':
                self._xmax = value
            elif key == 'ymin':
                self._ymin = value
            elif key == 'ymax':
                self._ymax = value
            else:
                raise ValueError("Invalid keyword argument: {}".format(key))

        # mesh parameters
        self._nx = nx
        self._ny = ny
        self._buff = buff
        self._setup()

    def _setup(self):
        # number of nodes including ghost nodes
        self._nxgc = self.nx + 2 * self.buff
        self._nygc = self.ny + 2 * self.buff

        # grid spacing
        self._dx = (self.xmax - self.xmin) / self.nx
        self._dy = (self.ymax - self.ymin) / self.ny

        # indices of the first and last nodes
        self._istart = self.buff
        self._iend = self.buff + self.nx - 1
        self._jstart = self.buff
        self._jend = self.buff + self.ny - 1

        # indices of the first and last ghost nodes
        self._istartg = 0
        self._iendg = self.buff + self.nx - 1 + self.buff
        self._jstartg = 0
        self._jendg = self.buff + self.ny - 1 + self.buff

        x = np.linspace(self.xmin, self.xmax, self.nxgc)
        y = np.linspace(self.ymin, self.ymax, self.nygc)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        self._mesh = xx*0
        self._x = x
        self._y = y
        self._xx = xx
        self._yy = yy
        return 

    @property
    def xmin(self):
        return self._xmin

    @property
    def xmax(self):
        return self._xmax

    @property
    def ymin(self):
        return self._ymin

    @property
    def ymax(self):
        return self._ymax

    @property
    def nx(self):
        return self._nx

    @property
    def ny(self):
        return self._ny

    @property
    def buff(self):
        return self._buff

    @property
    def nxgc(self):
        return self._nxgc

    @property
    def nygc(self):
        return self._nygc

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def istart(self):
        return self._istart

    @property
    def iend(self):
        return self._iend

    @property
    def jstart(self):
        return self._jstart

    @property
    def jend(self):
        return self._jend

    @property
    def istartg(self):
        return self._istartg

    @property
    def iendg(self):
        return self._iendg

    @property
    def jstartg(self):
        return self._jstartg

    @property
    def jendg(self):
        return self._jendg

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh):
        self._mesh = new_mesh

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def xx(self):
        return self._xx

    @property
    def yy(self):
        return self._yy

    @nx.setter
    def nx(self, value):
        print("Reset matrix to nx = {}".format(value))
        self._nx = value
        self._setup()

    @ny.setter
    def ny(self, value):
        print("Reset matrix to ny = {}".format(value))
        self._ny = value
        self._setup()

    @buff.setter
    def buff(self, value):
        print("Reset matrix to buff = {}".format(value))
        self._buff = value
        self._setup()
    
    @xmin.setter
    def xmin(self, value):
        print("Reset matrix to xmin = {}".format(value))
        self._xmin = value
        self._setup()
    
    @xmax.setter
    def xmax(self, value):
        print("Reset matrix to xmax = {}".format(value))
        self._xmax = value
        self._setup()

    @ymin.setter
    def ymin(self, value):
        print("Reset matrix to ymin = {}".format(value))
        self._ymin = value
        self._setup()

    @ymax.setter
    def ymax(self, value):
        print("Reset matrix to ymax = {}".format(value))
        self._ymax = value
        self._setup()

    @staticmethod
    def cal_error(mesh,new_mesh):
        return np.sqrt(np.sum((new_mesh-mesh)**2))

    @staticmethod
    @njit(parallel=True)
    def _jacobi(mesh,new_mesh,istart,iend,jstart,jend):
        for i in prange(istart,iend+1):
            for j in prange(jstart,jend+1):
                new_mesh[i,j] = 0.25*(mesh[i-1,j]+mesh[i+1,j]+mesh[i,j-1]+mesh[i,j+1])
        return new_mesh

    def jacobi(self,new_mesh,omega=None):
        new_mesh = self._jacobi(self.mesh,new_mesh,self.istart,self.iend,self.jstart,self.jend)
        return new_mesh, self.cal_error(self.mesh,new_mesh)

    @staticmethod
    @njit(parallel=True)
    def _gauss_seidel(new_mesh,istart,iend,jstart,jend):
        for i in prange(istart,iend+1):
            for j in prange(jstart,jend+1):
                new_mesh[i,j] = 0.25*(new_mesh[i-1,j]+new_mesh[i+1,j]+new_mesh[i,j-1]+new_mesh[i,j+1])
        return new_mesh

    def gauss_seidel(self,new_mesh,omega=None):
        new_mesh = self._gauss_seidel(new_mesh,self.istart,self.iend,self.jstart,self.jend)
        return new_mesh, self.cal_error(self.mesh,new_mesh)

    @staticmethod
    @njit(parallel=True)
    def _sor(new_mesh,istart,iend,jstart,jend,omega):
        for i in prange(istart,iend+1):
            for j in prange(jstart,jend+1):
                new_mesh[i,j] = (1-omega)*new_mesh[i,j]+omega*0.25*(new_mesh[i-1,j]+new_mesh[i+1,j]+new_mesh[i,j-1]+new_mesh[i,j+1])
        return new_mesh

    def sor(self,new_mesh,omega=1.5):
        new_mesh = self._sor(new_mesh,self.istart,self.iend,self.jstart,self.jend,omega)
        return new_mesh, self.cal_error(self.mesh,new_mesh)

if __name__=='__main__':
    mesh = Mesh2D(10, 10, 1)
    print("Testing ... nx=10, ny=10, buff =1")

