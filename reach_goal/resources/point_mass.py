import pybullet as p
import numpy as np

class PointMass:
    def __init__(self, dt, client):
        self.client = client
        self.dt = dt
        # Load the point mass
        point_mass_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0.84, 0, 1.0], physicsClientId=client)
        point_mass_collision = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, physicsClientId=client)
        self.point_mass_idx = p.createMultiBody(baseMass=0, 
            baseVisualShapeIndex=point_mass_visual, 
            baseCollisionShapeIndex=point_mass_collision,
            basePosition=[0.0, 0.0 ,0.1], 
            physicsClientId=client)

    def get_ids(self):
        return self.client, self.point_mass_idx

    def apply_action(self, action):
        # Set the desired velocity for the rigid body
        desiredVelocity = [action[0], action[1], 0]  # Velocity in x, y, z directions

        # Set the velocity of the rigid body
        p.resetBaseVelocity(self.point_mass_idx, linearVelocity=desiredVelocity)
    
    def get_observation(self):
        robotPos, robotOrn = p.getBasePositionAndOrientation(self.point_mass_idx)
        return (robotPos[0], robotPos[1])