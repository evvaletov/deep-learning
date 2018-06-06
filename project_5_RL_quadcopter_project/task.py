import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        #self.state_size = self.action_repeat * 12
        self.state_size0 = 2
        self.state_size = self.action_repeat * self.state_size0
        #self.action_low = [0,-10,10]
        #self.action_high = [900,10,10]
        #self.action_size = 3
        self.action_low = -1
        self.action_high = 1
        self.action_size = 1
        #self.bonus_reward = False
        
        # Goal
        self.target_pos = np.array(target_pos) if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        #reward = 1.-(abs(self.sim.pose[:3] - self.target_pos)).sum()-0.1*abs(np.mod(np.array(self.sim.pose[:3])*0.5/np.pi,[1,1,1])).sum()
        #reward = 1.-np.square(self.sim.pose[2] - self.target_pos[2])
        #reward = 50*np.exp(-0.01*np.square(self.sim.pose[2] - self.target_pos[2]))
        #print("reward",self.sim.v[2],self.sim.pose[2] - self.target_pos[2])
        #if self.sim.pose[2] - self.target_pos[2]>0:
        #    reward = 10.0/3-0.1*np.square(self.sim.pose[2] - self.target_pos[2])
        #else:
        #    reward = 10.0/3-0.1*np.square(self.sim.pose[2] - self.target_pos[2])
        reward = 3.0-np.abs(self.sim.pose[2] - self.target_pos[2])
        #reward -= 0.1*np.square(10*(self.sim.pose[0] - self.target_pos[0]))
        #reward -= 0.1*np.square(10*(self.sim.pose[1] - self.target_pos[1]))
        #if (self.sim.v[2]*(self.sim.pose[2] - self.target_pos[2])>0.1):
        #    if self.sim.v[2]<0:
        #        reward += 0.2*self.sim.v[2]*np.square(5*(self.sim.pose[2] - self.target_pos[2]))
        #    else:
        #        reward -= 0.2*self.sim.v[2]*np.square(5*(self.sim.pose[2] - self.target_pos[2]))
        #if (self.sim.v[1]*(self.sim.pose[1] - self.target_pos[1])>0.1):
        #    reward -= 0.4*np.square(self.sim.pose[0] - self.target_pos[0])
        #if (self.sim.v[0]*(self.sim.pose[0] - self.target_pos[0])>0.1):
        #    reward -= 0.4*np.square(self.sim.pose[0] - self.target_pos[0])
        #if (self.sim.pose[2]<self.target_pos[2]*1.1) and (self.sim.pose[2]>self.target_pos[2]*0.9):
        #    reward += 10000
        #if (self.sim.pose[0]<self.target_pos[0]*1.01) and (self.sim.pose[0]>self.target_pos[0]*0.99) and (self.sim.time>4.5):
        #    reward += 10
        #if (self.sim.pose[1]<self.target_pos[1]*1.01) and (self.sim.pose[1]>self.target_pos[1]*0.99) and (self.sim.time>4.5):
        #    reward += 10
        #reward = min(reward, 50)
        #reward = max(reward, -50)
        #print("reward",reward)
        if (self.sim.v[2]<0) and (self.sim.pose[2]<self.target_pos[2]/2):
            reward += 5*self.sim.v[2] - 20/np.sqrt(np.square(self.sim.time)+np.square(self.sim.pose[2]))
        #    if (self.sim.pose[2]<self.target_pos[2]/10):
        #        reward += 10*self.sim.v[2]
        if (self.sim.v[2]>0) and (self.sim.pose[2]>1.5*self.target_pos[2]):
            reward -= 5*self.sim.v[2]
        if done:
            if (self.sim.time<self.sim.runtime):
                reward -= 10.0+3*np.abs(self.target_pos[2])+3*np.abs(self.sim.time-self.sim.runtime)
            else:
                if (self.sim.pose[2]<=self.target_pos[2]-1):
                    reward += 10.0-3*np.abs(self.sim.pose[2] - self.target_pos[2])
                elif (self.sim.pose[2]<self.target_pos[2]+1) and (self.sim.pose[2]>self.target_pos[2]-1):
                    reward += 50
                else:
                    reward += 10.0-np.abs(self.sim.pose[2] - self.target_pos[2])/2
        if (self.sim.pose[2]<self.target_pos[2]+1) and (self.sim.pose[2]>self.target_pos[2]-1) and (self.sim.time>self.sim.runtime*0.4):
            reward += 50
            #self.bonus_reward = True
        return reward

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []
        z_speed = 450+100*action[0]
        #z_speed = 403
        #rotor_speeds = np.array([z_speed,z_speed,z_speed,z_speed]) + np.array([action[1],-action[1],0,0]) + np.array([0,0,-action[2],action[2]])
        rotor_speeds = np.array([z_speed,z_speed,z_speed,z_speed])
        #print("Rotor speeds",rotor_speeds)
        # 1: x+, x-, y-, y+
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(done) 
            #state_all.append(np.concatenate((self.sim.pose,self.sim.v,self.sim.angular_v)))
            #state_all.append(self.sim.pose)
            #state_all.append(np.concatenate((self.sim.pose,self.sim.v)))
            state_all.append([self.sim.pose[2],self.sim.v[2]])
        next_state = np.concatenate(state_all)
        #next_state = [self.sim.pose[2],self.sim.v[2]]
        #next_state = [self.sim.pose[2]]
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate([np.concatenate((self.sim.pose,self.sim.v,self.sim.angular_v))] * self.action_repeat) 
        #state = np.concatenate([self.sim.pose] * self.action_repeat) 
        #state = np.concatenate([np.concatenate((self.sim.pose,self.sim.v))] * self.action_repeat) 
        state = [self.sim.pose[2],self.sim.v[2]]*self.action_repeat
        #self.bonus_reward = False
        #state = [self.sim.pose[2],self.sim.v[2]] 
        #state = [self.sim.pose[2]] 
        return state