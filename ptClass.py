import torch
from vmas.simulator.core import Agent
from vmas.simulator.utils import Color

class Tower(Agent):
    def __init__(self, name, shape, control_radius=4, color=Color.GREEN, **kwargs):
        super().__init__(name, shape, color=color, **kwargs)
        movable=False
        rotatable=False
        collide=False
        self.control_radius = control_radius

    #distance between tower and others(plane ot other towers)
    def distance(self, otherpos):
        #np.linalg.norm(np.array(self.position) - np.array(otherpos))
        #((otherpos[0] - self.position[0]) ** 2 + (otherpos[1] - self.position[1]) ** 2) ** 0.5
        distance = (otherpos - self.state.pos).norm()
        return distance

    #detect if plane is in tower's area
    def isinarea(self, planepos):
        dis = self.distance(planepos)
        if dis < self.control_radius:
            return True
        else:
            return False


    # Update all towers' positions when (0, 0)'s tower change
    def updatepos(self, newtower, towers):
        shift = newtower.state.pos
        for t in towers:
            t.state.pos = t.state.pos - shift


class Plane(Agent):
    def __init__(self, name, shape, color=Color.BLUE, **kwargs):
        super().__init__(name, shape, color=color, **kwargs)
        #self.listensignal = []
        movable=True
        rotatable=True
        collide=False
        self.listening_to = None


    # Update plane position when (0, 0)'s tower change
    def update_position(self, tower):
        shift = tower.state.pos
        self.state.pos = self.state.pos - shift

    # distance between plane and tower
    def distance(self, towerpos):
        dis = (towerpos - self.state.pos).norm()
        return dis

    #plane listen to the closest tower
    def listen(self, towers):
        abletower=[]
        for t in towers:
            if t.isinarea(self.state.pos):
                abletower.append(t)


        min = float('inf')
        closesttower = None
        for t in abletower:
            dis = self.distance(t.state.pos)
            if dis < min:
                min = dis
                closesttower = t

        self.listening_to = closesttower

