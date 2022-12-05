import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numba import jit, njit, prange, set_num_threads, get_num_threads

"""

This program solve 3D direct N-particles simulations 
under gravitational forces. 

This file contains two classes:

1) Particles: describes the particle properties
2) NbodySimulation: describes the simulation

Usage:

    Step 1: import necessary classes

    from nbody import Particles, NbodySimulation

    Step 2: Write your own initialization function

    
        def initialize(particles:Particles):
            ....
            ....
            particles.set_masses(mass)
            particles.set_positions(pos)
            particles.set_velocities(vel)
            particles.set_accelerations(acc)

            return particles

    Step 3: Initialize your particles.

        particles = Particles(N=100)
        initialize(particles)


    Step 4: Initial, setup and start the simulation

        simulation = Simulation(particles)
        simulation.setip(...)
        simulation.evolve(dt=0.001, tmax=10)


Author: Kuo-Chuan Pan, NTHU 2022.10.30
For the course, computational physics lab

"""
set_num_threads(get_num_threads())


class Particles:
    """

    The Particles class handle all particle properties

    for the N-body simulation. 

    """

    def __init__(self, N: int = 100):
        """
        Prepare memories for N particles

        :param N: number of particles.

        By default: particle properties include:
                nparticles: int. number of particles
                _masses: (N,1) mass of each particle
                _positions:  (N,3) x,y,z positions of each particle
                _velocities:  (N,3) vx, vy, vz velocities of each particle
                _accelerations:  (N,3) ax, ay, az accelerations of each partciel
                _tags:  (N)   tag of each particle
                _time: float. the simulation time 

        """
        self.nparticles = N
        self._masses = np.ones((N, 1))
        self._positions = np.zeros((N, 3))
        self._velocities = np.zeros((N, 3))
        self._accelerations = np.zeros((N, 3))
        self._tags = np.linspace(1, N, N)
        self._time = 0.0
        return

    @property
    def masses(self):
        return self._masses

    @property
    def positions(self):
        return self._positions

    @property
    def velocities(self):
        return self._velocities

    @property
    def accelerations(self):
        return self._accelerations

    @property
    def tags(self):
        return self._tags

    @property
    def time(self):
        return self._time

    @masses.setter
    def masses(self, masses):
        for i in range(self.nparticles):
            self._masses[i] = masses[i]

    @positions.setter
    def positions(self, pos):
        for i in range(self.nparticles):
            for j in range(3):
                self._positions[i, j] = pos[i, j]

    @velocities.setter
    def velocities(self, vel):
        for i in range(self.nparticles):
            for j in range(3):
                self._velocities[i, j] = vel[i, j]

    @accelerations.setter
    def accelerations(self, acc):
        for i in range(self.nparticles):
            for j in range(3):
                self._accelerations[i, j] = acc[i, j]

    @tags.setter
    def tags(self, tags):
        for i in range(self.nparticles):
            self._tags[i] = tags[i]

    @time.setter
    def time(self, t):
        self._time = t

    def output(self, fn, time):
        """
        Write simulation data into a file named "fn"


        """
        mass = self._masses
        pos = self._positions
        vel = self._velocities
        acc = self._accelerations
        tag = self._tags
        header = """
                ----------------------------------------------------
                Data from a 3D direct N-body simulation. 

                rows are i-particle; 
                coumns are :tag, mass, x ,y, z, vx, vy, vz, ax, ay, az

                NTHU, Computational Physics Lab

                ----------------------------------------------------
                """
        header += "Time = {}".format(time)
        np.savetxt(fn, (tag[:], mass[:, 0], pos[:, 0], pos[:, 1], pos[:, 2],
                        vel[:, 0], vel[:, 1], vel[:, 2],
                        acc[:, 0], acc[:, 1], acc[:, 2]), header=header)

        return


class NbodySimulation:
    """

    The N-body Simulation class.

    """

    def __init__(self, particles: Particles):
        """
        Initialize the N-body simulation with given Particles.

        :param particles: A Particles class.  

        """

        # store the particle information
        self.nparticles = particles.nparticles
        self.particles = particles

        # Store physical information
        self.time = 0.0  # simulation time

        # Set the default numerical schemes and parameters
        self.setup()

        return

    def setup(self, G=1,
              rsoft=0.01,
              method="RK4",
              io_freq=10,
              io_title="particles",
              io_screen=True,
              visualized=False):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4"

        :param io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output. 
        :param io_title: the output header
        :param io_screen: print message on screen or not.
        :param visualized: on the fly visualization or not. 

        """
        self.G = G
        self.rsoft = rsoft
        self.method = method
        self.io_freq = io_freq
        self.io_title = io_title
        self.io_screen = io_screen
        self.visualized = visualized
        return

    def evolve(self, dt: float = 0.01, tmax: float = 1):
        """

        Start to evolve the system

        :param dt: time step
        :param tmax: the finial time

        """
        method = self.method
        if method == "Euler":
            _update_particles = self._update_particles_euler
        elif method == "RK2":
            _update_particles = self._update_particles_rk2
        elif method == "RK4":
            _update_particles = self._update_particles_rk4
        else:
            print("No such update meothd", method)
            quit()

        # prepare an output folder for lateron output
        io_folder = "data_"+self.io_title
        Path(io_folder).mkdir(parents=True, exist_ok=True)
        # ====================================================
        #
        # The main loop of the simulation
        #
        # =====================================================
        i = 0
        record = 0
        while self.time < tmax:
            if((self.io_freq>0 and i%self.io_freq==0) or self.io_freq == 0):
                print(i)
                fn = io_folder+"/"+str(record).zfill(5)+".txt"
                self.particles.output(fn, self.time)
                record += 1
                if(self.io_screen):
                    print("Data saved at time = ", self.time)

            self.particles.accelerations = self._calculate_acceleration(self.particles.masses, self.particles.positions)
            self.particles = _update_particles(dt, self.particles)
            if(self.visualized):
                self._visualize()
            self.time += dt
            i+=1
        print("Done!")
        return

    @staticmethod
    @njit(parallel=True)
    def acc_loop(acc,N,rsoft,G,mass,pos):
        for i in prange(N):
                for j in prange(N):
                    if j!=i:
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dz = pos[i, 2] - pos[j, 2]
                        r = (dx**2 + dy**2 +dz**2 + rsoft**2)**0.5
                        acc[i, 0] += -G * mass[j,0] * dx / r**3
                        acc[i, 1] += -G * mass[j,0] * dy / r**3
                        acc[i, 2] += -G * mass[j,0] * dz / r**3
        return acc

    def _calculate_acceleration(self, mass, pos):
        """
        Calculate the acceleration.
        """
        acc = np.zeros_like(self.particles.accelerations)
        return self.acc_loop(acc,self.nparticles,self.rsoft,self.G,mass,pos)

    def _update_particles_euler(self, dt, particles: Particles):
        """ 
        Update the particles using Euler method
        """
        particles.positions += particles.velocities * dt
        particles.velocities += particles.accelerations * dt
        return particles

    def _update_particles_rk2(self, dt, particles: Particles):
        """
        Update the particles using RK2 method
        """
        acc1 = particles.accelerations
        pos1 = particles.positions + particles.velocities * dt
        vel1 = particles.velocities + acc1 * dt
        acc2 = self._calculate_acceleration(particles.masses, pos1)
        particles.positions += 0.5 * dt * (particles.velocities + vel1)
        particles.velocities += 0.5 * dt * (acc1 + acc2)
        return particles

    def _update_particles_rk4(self, dt, particles: Particles):
        """
        Update the particles using RK4 method
        """
        acc1 = particles.accelerations
        pos1 = particles.positions + particles.velocities * dt
        vel1 = particles.velocities + acc1 * dt
        acc2 = self._calculate_acceleration(particles.masses, pos1)
        pos2 = particles.positions + vel1 * dt/2
        vel2 = particles.velocities + acc2 * dt/2
        acc3 = self._calculate_acceleration(particles.masses, pos2)
        pos3 = particles.positions + vel2 * dt/2
        vel3 = particles.velocities + acc3 * dt/2
        acc4 = self._calculate_acceleration(particles.masses, pos3)
        particles.positions += dt / 6.0 * (particles.velocities + 2.0 * vel1 + 2.0 * vel2 + vel3)
        particles.velocities += dt / 6.0 * (acc1 + 2.0 * acc2 + 2.0 * acc3 + acc4)
        return particles

    def _visualize(self):
        # TODO:
        return

if __name__ == '__main__':

    # test Particles() here
    particles = Particles(N=2)
    particles.positions = np.array([[0, 0,0], [1., 0,0]])
    particles.velocities = np.array([[0, 0,0], [0, 1.,0]])
    # test NbodySimulation(particles) here
    sim = NbodySimulation(particles=particles)
    sim.setup(method="Euler", io_freq=10, io_title="test")
    sim.evolve(dt=0.01, tmax=10)
    print("Done")
