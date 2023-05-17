import pybullet as p
import os

class Goal:
    def __init__(self, client, pos):
        # Add the goal to the environment
        goal_position = [pos[0], pos[1], 0.1]
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1.0], physicsClientId=client)
        sphere_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual, basePosition=goal_position, physicsClientId=client)