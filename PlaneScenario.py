from vmas.simulator.scenario import BaseScenario
from vmas.simulator.core import Agent, Landmark, World, Sphere
from vmas.simulator.utils import Color
from vmas.simulator.sensors import Lidar
from vmas import render_interactively
import torch
from ptClass import Plane, Tower

class PlaneScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 3)
        n_landmarks = kwargs.get("n_landmarks", 2)
        #self.pos_range = kwargs.get("pos_range", 10)

        # Make world
        world = World(batch_dim, device,
            x_semidim=None,
            y_semidim=None,
        )

        # Add agents
        plane = Plane(
            name="plane",
            shape=Sphere(radius=0.1),
            collide=False,

        )
        world.add_agent(plane)
        towerA = Tower(
            name="towerA",
            shape=Sphere(radius=0.2),
            #obs_range = 5,
            movable=False,
            rotatable=False,
            collide=False,
        )
        world.add_agent(towerA)
        towerB = Tower(
            name="towerB",
            shape=Sphere(radius=0.2),
            #obs_range = 5,
            movable=False,
            rotatable=False,
            collide=False,
        )
        world.add_agent(towerB)

        # Add landmarks
        start = Landmark(
            name="start",
            collide=False,
            shape=Sphere(radius=0.2),
            color=Color.GRAY,
            movable=False,
        )
        world.add_landmark(start)
        end = Landmark(
            name="end",
            collide=False,
            shape=Sphere(radius=0.2),
            color=Color.GRAY,
            movable=False,
        )
        world.add_landmark(end)

        return world

    def reset_world_at(self, env_index: int = None):
        for i, agent in enumerate(self.world.agents):
            if agent.name == "plane":
                agent.set_pos(
                    torch.tensor(
                        [0, 0], dtype=torch.float32, device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                agent.set_vel(
                    torch.tensor(
                        [4.7, 2],  dtype=torch.float32, device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                agent.state.pos += agent.state.vel
            if agent.name == "towerA":
                agent.set_pos(
                    torch.tensor(
                        [0.5, 0], dtype=torch.float32, device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            if agent.name == "towerB":
                agent.set_pos(
                    torch.tensor(
                        [7, 0], dtype=torch.float32, device=self.world.device,
                    ),
                    batch_index=env_index,
                )

        for i, landmark in enumerate(self.world.landmarks):
            if landmark.name == "start":
                landmark.set_pos(
                    torch.tensor(
                        [0, 0], dtype=torch.float32, device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            if landmark.name == "end":
                landmark.set_pos(
                    torch.tensor(
                        [6, 0], dtype=torch.float32, device=self.world.device,
                    ),
                    batch_index=env_index,
                )


    def observation(self, agent: Agent):
        obs = []
        planepos = self.world.agents[0].state.pos
        planevel = self.world.agents[0].state.vel
        obs.append(planepos)
        obs.append(planevel)

        for landmark in self.world.landmarks:
            relative_pos = landmark.state.pos - planepos
            obs.append(relative_pos)

        for tower in [self.world.agents[1], self.world.agents[2]]:
            ifin = torch.tensor([tower.isinarea(planepos)])
            if tower.isinarea(planepos):
                center = tower.state.pos
                relative_pos = center - planepos
            if tower.isinarea(planepos)==False:
                relative_pos = tower.state.pos - planepos
            obs.append(ifin.unsqueeze(0).expand(self.world.batch_dim, -1))
            obs.append(relative_pos)

        # Concatenate all components into a single observation vector
        return torch.cat(obs, dim=-1)

    def reward(self, agent: Agent):
        rewards = torch.zeros(self.world.batch_dim, device=self.world.device)

        # end place reward
        distance_to_end = torch.linalg.vector_norm(self.world.landmarks[1].state.pos - self.world.agents[0].state.pos)
        end_place_reward = 10-distance_to_end

        # If the plane listening to the closest tower
        distance_to_A = torch.linalg.vector_norm(self.world.agents[0].state.pos - self.world.agents[1].state.pos)
        distance_to_B = torch.linalg.vector_norm(self.world.agents[0].state.pos - self.world.agents[2].state.pos)
        righttower = None
        if distance_to_A < distance_to_B:
            righttower=self.world.agents[1]
        elif distance_to_A >= distance_to_B:
            righttower=self.world.agents[2]

        self.world.agents[0].listen([self.world.agents[1], self.world.agents[2]])
        if self.world.agents[0].listening_to == righttower:
            listen_reward = 5
        else:
            listen_reward = 0

        all = end_place_reward+listen_reward
        rewards.fill_(all)
        return rewards

    def done(self):
        # Check if the plane arrives in
        distance_to_end = torch.linalg.vector_norm(self.world.landmarks[1].state.pos - self.world.agents[0].state.pos)
        # reach if within 0.1 distance of the end
        ifend = distance_to_end <= 0.1
        alldone = ifend

        return torch.tensor(alldone, device=self.world.device).expand(self.world.batch_dim)
    

