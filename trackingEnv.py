import os
import pickle
import time

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import airsim
from scipy.integrate import odeint
from scipy.spatial.transform.rotation import Rotation as R

import wandb
import cv2
from scipy.interpolate import CubicSpline
from unrealEnv import UnrealEnv as UE
import utilities
def reward_tracking_vision(x, y, z,
                           u=np.zeros((4,)),
                           v=np.zeros((4,)),
                           optimal_x=4,
                           max_x_dist=7.0,
                           min_x_dist=1.0,
                           exp=(1 / 3),
                           alpha=0.0,
                           beta=0.0,
                           gamma=0.2,
                           segma=0.2,
                           max_steps=400.0,
                           dt = 0.05,
                           collision=False):
    done = False

    y_ang = np.arctan(y / x)
    z_ang = np.arctan(z / x)

    y_error = abs(y_ang / (np.pi / 4))
    z_error = abs(z_ang / (np.pi / 4))
    x_error = abs(x - optimal_x)

    z_rew = max(0, 1 - z_error)
    y_rew = max(0, 1 - y_error)
    x_rew = max(0, 1 - x_error)

    vel_penalty = np.linalg.norm(v) / (1 + np.linalg.norm(v))
    u_penalty = np.linalg.norm(u) / (1 + np.linalg.norm(u))


    reward_track = (x_rew * y_rew * z_rew) ** exp

    reward = (reward_track - alpha * vel_penalty - beta * u_penalty )

    if abs(np.linalg.norm(np.array([x,y,z]))) > max_x_dist or abs((np.linalg.norm(np.array([x,y,z])))) < min_x_dist \
            or collision :
        done = True
        reward = -10

    return reward, done, reward_track, x_error, y_error, z_error

def drone_dyn(X, t, g, m, w, f):
    X = np.expand_dims(X, axis=1)
    # Variables and Parameters
    zeta = np.array([0, 0, 1]).reshape(3, 1)
    gv = np.array([0, 0, -1]).reshape(3, 1) * g
    p = X[0: 3, 0]
    v = X[3: 6, 0]
    R = X[6: 15].reshape(3, 3)

    # Drone Dynamics
    dp = v
    dv = np.dot(R, zeta) * f / m + gv
    dR = np.dot(R, sKw(w))

    #
    dX = np.concatenate((dp.reshape(-1, 1), dv, dR.reshape(-1, 1)), axis=0).squeeze()
    return dX
def sKw(x):
    Y = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]], dtype=np.float64)
    return Y

class TrackingEnv(gym.Env):
    def __init__(self,
                 rank = 1,
                 policy="policy",
                 optimal_x=4,
                 max_x_dist = 0.5,
                 min_x_dist = 0.1,
                 dt=0.01,
                 episode_time=800,
                 stack_length=3,
                 vect_size = 3,
                 action_limit=4,
                 action_dim=4,
                 screen_height=100,
                 screen_width=100,
                 obs_type = '',
                 random_pose_init = True,
                 alpha=0.4,
                 beta=0.4,
                 gamma=0.2,
                 segma=0.2,
                 WandB=False,
                 debug = True):


        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.action_limit = action_limit
        self.action_dim = action_dim
        self.action_space = gym.spaces.Box(-self.action_limit, self.action_limit, shape=(self.action_dim,),
                                           dtype=np.float32)
        self.stack_length = stack_length
        self.WandB = WandB
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.image_shape = (screen_height,screen_width,3)
        self.vect_size = vect_size  # Size of latent vector
        self.obs_type = obs_type
        #########################################################
        print("df_env")

        if self.obs_type == "Event_rep_vector":
            actor_obs_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.stack_length * self.vect_size,), dtype=np.float32
            )
        elif self.obs_type == "event_img":
            actor_obs_space = gym.spaces.Box(low=0.0, high=1.0,
                                             shape=(self.stack_length,) + (2, self.screen_height, self.screen_width,),
                                             dtype=np.float32)

        elif self.obs_type == "state_vector" :
            actor_obs_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.stack_length * self.vect_size,), dtype=np.float32
            )
        elif self.obs_type == "RGB":
            actor_obs_space = gym.spaces.Box(low=0.0, high=1.0,
                                             shape=(self.stack_length,) + (3, self.screen_height, self.screen_width,),
                                             dtype=np.float32)





        critic_obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'actor': actor_obs_space,
            'critic': critic_obs_space
        })

        self.optimal_x = optimal_x
        self.max_x_dist = max_x_dist
        self.min_x_dist = min_x_dist

        # REWARD PARAMETERS
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.segma = segma
        self.random_pose_init = random_pose_init

        # RL TRAINING
        # Variables
        self.state = None
        self.reward = 0
        self.done = 0
        self.episode_steps = 0
        self.info = {}

        # data history
        self.stack_obs = None

        # EPISODE
        self.dt = dt  # Sampling time[s] 1
        self.episode_time = episode_time  # s
        self.max_episode_steps = int(self.episode_time / self.dt)
        self.rank = rank
        self.stop_step = random.randint(0, self.max_episode_steps)
        self.stop_duration = 10 / self.dt

        # SIMULATION PARAMETERS
        self.g = 9.8  # Gravitational Acceleration[m / s ^ 2]
        self.Tin = 0  # Initial time[s]
        self.T_target = 0.0
        self.Ts_target = dt
        self.m_to_airsim = 10        # DRONE PARAMETERS (1 airsim --> 0.1m )
        self.m = 1  # # Tracker Mass[kg]
        self.Xin = None
        self.fk = self.m * self.g

        # TARGET MOVEMENT
        self.move_target = True
        self.target_follow_tarj = False
        self.pr0 = np.array([self.optimal_x, 0.0, 0.0])
        self.prout = np.array([0.0, 0.0, 0.0])
        self.vrout = np.array([0.0, 0.0, 0.0])
        self.arout = np.array([0.0, 0.0, 0.0])
        self.Rrout = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
        self.a1, self.a2, self.a3 = None, None, None
        self.phi1, self.phi2, self.phi3 = None, None, None
        self.ws1, self.ws2, self.ws3 = None, None, None
        self.scale_factor = 0
        self.reset_target()

        if self.target_follow_tarj :
            directory = r"C:\Users\c23liard\Desktop\Ala_Workspace\Projects\AirSim_codes\myscripts\cave_traj"
            filename = "positions.pkl"
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as file:
                positions_in_UE = pickle.load(file)
            #add initial position
            positions_in_UE = [(self.pr0*self.m_to_airsim).tolist()] + positions_in_UE
            positions_in_m = [
                                [pos[0] / self.m_to_airsim, pos[1] / self.m_to_airsim, pos[2] / -self.m_to_airsim]
                                for pos in positions_in_UE
                            ]
            self.Target_trajectory_in_m = self.interpolate_positions(positions_in_m, self.max_episode_steps)

        # Unreal engine instance
        self.UE = UE(rank = rank, client=self.client, optimal_x=self.optimal_x, dt=self.dt, max_episode_steps= self.max_episode_steps
                     , screen_height=self.screen_height, screen_width=self.screen_width, random_pose_init=self.random_pose_init,
                     stack_length = self.stack_length, vect_size= self.vect_size, obs_type=self.obs_type, debug = debug)

        # RENDER trajectory
        self.fig = None
        self.ax = None
        self.axs = []

        self.tracker_position_x = []
        self.tracker_position_y = []
        self.tracker_position_z = []

        self.target_position_x = []
        self.target_position_y = []
        self.target_position_z = []

    def interpolate_positions(self,positions, num_points):
        # Filter out None values from the positions list
        filtered_positions = [pos for pos in positions if pos is not None]
        # Extract x, y, and z components
        import numpy as np
        xs = np.array([pos[0] for pos in filtered_positions])
        ys = np.array([pos[1] for pos in filtered_positions])
        zs = np.array([pos[2] for pos in filtered_positions])

        # Generate the original indices
        original_indices = np.linspace(0, len(filtered_positions) - 1, num=len(filtered_positions))

        # Generate the new indices for interpolation
        new_indices = np.linspace(0, len(filtered_positions) - 1, num=num_points)

        # Interpolate x, y, and z components
        interp_xs = np.interp(new_indices, original_indices, xs)
        interp_ys = np.interp(new_indices, original_indices, ys)
        interp_zs = np.interp(new_indices, original_indices, zs)

        # Combine interpolated x, y, and z components
        interpolated_positions = np.vstack((interp_xs, interp_ys, interp_zs)).T
        return interpolated_positions

    def reset_target(self, p=None, rand=False):
        self.T_target = 0
        if self.move_target or self.target_follow_tarj:
            self.a1 = random.uniform(-0.75, 1.25)
            self.a2 = random.uniform(-0.75, 1.25)
            self.a3 = random.uniform(-0.75, 0.75)

            if 0 <= self.a1 <= 0.75:self.a1 = 0.75
            elif -0.75 <= self.a1 < 0:self.a1 = -0.75
            if 0 <= self.a2 <= 0.75: self.a2 = 0.75
            elif -0.75 <= self.a2 < 0:self.a2 = -0.75
            if 0 <= self.a3 <= 0.75:self.a3 = 0.75
            elif -0.75 <= self.a3 < 0:self.a3 = -0.75

            k1 = 6 + random.random() * 6
            k2 = 6 + random.random() * 6
            k3 = 6 + random.random() * 6

            f1 = 1 / (k1 * abs(self.a1))
            f2 = 1 / (k2 * abs(self.a2))
            f3 = 1 / (k3 * abs(self.a3))

            self.phi1 = -np.pi / 2 + random.random() * np.pi / 2
            self.phi2 = -np.pi / 2 + random.random() * np.pi / 2
            self.phi3 = -np.pi / 2 + random.random() * np.pi / 2

            self.ws1 = 2 * np.pi * f1
            self.ws2 = 2 * np.pi * f2
            self.ws3 = 2 * np.pi * f3

            if p is not None:
                if rand:
                    fov = np.pi / 4
                    xdis = random.random() + self.optimal_x + 0.1
                    yang = random.random() * fov - fov / 2
                    zang = random.random() * fov - fov / 2
                    ydis = xdis * np.tan(yang)
                    zdis = xdis * np.tan(zang)

                    self.pr0 = np.array([p[0] + xdis, p[1] + ydis, p[2] + zdis])
                else:
                    self.pr0 = np.array([p[0] + self.optimal_x, p[1], p[2]])
            else:
                self.pr0 = np.array([self.optimal_x, 0.0, 0.0])

        self.prout = self.pr0
        self.vrout = np.array([0.0, 0.0, 0.0])
        self.arout = np.array([0.0, 0.0, 0.0])
        self.Rrout = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
    def step(self, u):
        self.info = {}
        self.episode_steps += 1

        #move target
        if self.move_target and (self.episode_steps <= self.stop_step or self.episode_steps > (self.stop_step + self.stop_duration)):
            self.prout = np.array([self.a1 * np.sin(self.phi1 + self.ws1 * self.T_target) - self.a1 * np.sin(self.phi1) + self.pr0[0],
                                   self.a2 * np.sin(self.phi2 + self.ws2 * self.T_target) - self.a2 * np.sin(self.phi2) + self.pr0[1],
                                   self.a3 * np.sin(self.phi3 + self.ws3 * self.T_target) - self.a3 * np.sin(self.phi3) + self.pr0[2]])
            self.vrout = np.array([self.a1 * self.ws1 * np.cos(self.phi1 + self.T_target * self.ws1),
                                   self.a2 * self.ws2 * np.cos(self.phi2 + self.T_target * self.ws2),
                                   self.a3 * self.ws3 * np.cos(self.phi3 + self.T_target * self.ws3)])
            self.arout = np.array([-self.a1 * self.ws1 ** 2 * np.sin(self.phi1 + self.T_target * self.ws1),
                                   -self.a2 * self.ws2 ** 2 * np.sin(self.phi2 + self.T_target * self.ws2),
                                   -self.a3 * self.ws3 ** 2 * np.sin(self.phi3 + self.T_target * self.ws3)])

            self.scale_factor =min(self.scale_factor + 0.005, 1)
            self.prout = self.pr0 + self.scale_factor * (self.prout - self.pr0)
            self.vrout = self.scale_factor * self.vrout

            self.T_target += self.Ts_target
        elif not self.move_target and not self.target_follow_tarj :
            self.scale_factor = 0
            self.vrout = np.array([0, 0, 0])
            self.arout = np.array([0, 0, 0])
        elif self.target_follow_tarj :
            pos_in_m = self.Target_trajectory_in_m[self.episode_steps]
            prev_pos_in_m= self.Target_trajectory_in_m[self.episode_steps-1]
            self.prout = np.array(pos_in_m)
            self.vrout = (self.prout - np.array(prev_pos_in_m) )/ self.dt
            self.arout = np.array([0,0,0])

        # Get Tracker's commands
        u = u.reshape(self.action_dim, )
        w = u[:3]
        f = (u[3] * 5 + 20.2) / 2
        self.fk = f
        # Integrate dynamics
        time_interval = [self.Tin, self.Tin + self.dt]
        Xout = odeint(drone_dyn, self.Xin.squeeze(), time_interval, args=(self.g, self.m, w, self.fk))  # X, t, g, m, w, f
        Xout = Xout[-1, :].T
        Tout = time_interval[-1]

        # Tracker output variables
        pout = Xout[0: 3]
        vout = Xout[3: 6]
        Rout = Xout[6: 15].reshape(3, 3)

        zeta = np.array([0, 0, 1]).reshape(3, 1)
        gv = np.array([0, 0, -1]).reshape(3, 1) * self.g
        aout = (np.dot(Rout, zeta) * (self.fk / self.m) + gv).reshape(3, )

        # X Y Z of the target wrt to the tracker at time Tout
        x, y, z = np.dot(np.dot(np.array([1, 0, 0]), Rout.T), self.prout - pout), \
                  np.dot(np.dot(np.array([0, 1, 0]), Rout.T), self.prout - pout), \
                  np.dot(np.dot(np.array([0, 0, 1]), Rout.T), self.prout - pout)

        v_x, v_y, v_z = np.dot(np.dot(np.array([1, 0, 0]), Rout.T), self.vrout - vout), \
                        np.dot(np.dot(np.array([0, 1, 0]), Rout.T), self.vrout - vout), \
                        np.dot(np.dot(np.array([0, 0, 1]), Rout.T), self.vrout - vout)

        a_x, a_y, a_z = np.dot(np.dot(np.array([1, 0, 0]), Rout.T), self.arout - aout), \
                        np.dot(np.dot(np.array([0, 1, 0]), Rout.T), self.arout - aout), \
                        np.dot(np.dot(np.array([0, 0, 1]), Rout.T), self.arout - aout)


        # Move target and tracker in Unreal engine
        # Target
        p_target = self.prout.reshape(3,)
        R_target = self.Rrout
        qx, qy, qz, qw = R.from_matrix(R_target).as_quat()
        target_new_pose = [(p_target[0]-self.optimal_x)*self.m_to_airsim, p_target[1]*self.m_to_airsim, -p_target[2]*self.m_to_airsim, -qx, -qy, qz, qw]
        self.UE.set_target_pose(target_new_pose)
        # Tracker
        p_tracker = self.Xin[0: 3].reshape(3,)
        R_tracker = self.Xin[6: 15].reshape(3, 3)
        qx, qy, qz, qw = R.from_matrix(R_tracker).as_quat()
        tracker_new_pose = [p_tracker[0]*self.m_to_airsim,p_tracker[1]*self.m_to_airsim,-p_tracker[2]*self.m_to_airsim, -qx, -qy, qz, qw]
        self.UE.set_tracker_pose(tracker_new_pose)

        #time.sleep(self.dt)  #for a real time visualisation
        # Both the tracker and the target reached there new state we get the new position of both of them

        # render updating the trajectories
        self.tracker_position_x.append(p_tracker[0])
        self.tracker_position_y.append(p_tracker[1])
        self.tracker_position_z.append(p_tracker[2])

        self.target_position_x.append(p_target[0])
        self.target_position_y.append(p_target[1])
        self.target_position_z.append(p_target[2])


        # Get states

        critic_obs = np.array([x-self.optimal_x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]).reshape((1, 9))
        if self.obs_type == 'state_vector' :
            actor_current_obs = np.array([x - self.optimal_x, y, z]).reshape((1, 3))
        else :
            #get obs from simulation
            actor_current_obs = self.UE.get_obs()

        #actor_current_obs = self.augment_image(actor_current_obs)
        # Update feat_vect history
        if self.obs_type == 'RGB':
            actor_current_obs = np.expand_dims(actor_current_obs, axis=0)
            if self.stack_obs is None:
                self.stack_obs = [actor_current_obs] * self.stack_length
            self.stack_obs.append(actor_current_obs)
            self.stack_obs.pop(0)
            actor_obs = np.concatenate(self.stack_obs, axis=0)

        elif self.obs_type == 'Event_rep_vector':
            if self.stack_obs is None:
                self.stack_obs  = [actor_current_obs] * self.stack_length
            self.stack_obs.append(actor_current_obs)
            self.stack_obs.pop(0)
            actor_obs = np.concatenate(self.stack_obs, axis=1)

        elif self.obs_type == "event_img":
            actor_current_obs = np.expand_dims(actor_current_obs, axis=0)
            if self.stack_obs is None:
                self.stack_obs = [actor_current_obs] * self.stack_length
            self.stack_obs.append(actor_current_obs)
            self.stack_obs.pop(0)
            actor_obs = np.concatenate(self.stack_obs, axis=0)

        elif  self.obs_type == "state_vector" :
            if self.stack_obs is None:
                self.stack_obs  = [actor_current_obs] * self.stack_length
            self.stack_obs.append(actor_current_obs)
            self.stack_obs.pop(0)
            actor_obs = np.concatenate(self.stack_obs, axis=1)


        self.state = {
            'actor': actor_obs,
            'critic': critic_obs
        }
        # SubProcVecEnv Fix
        self.state['critic'] = self.state['critic'].flatten()
        if self.obs_type == 'Event_rep_vector' or self.obs_type == "state_vector"  :
            self.state['actor'] = self.state['actor'].flatten()
        # the shape of the obs shape is (1,3*vector_size) the feats_extractor will flatten it to --> (3*vect_size,)
        # the SubProcVecEnv expect an observation in shape (3*vect_size,) then it broadcast it into shape (n_envs,3*vect_size)


        # REWARD AND DONE
        # self.collision_info = airsim.types.CollisionInfo()
        self.reward, self.done, reward_track, x_error, y_error, z_error = reward_tracking_vision(x,y,z,
                                                                                                 v=np.array([v_x, v_y, v_z]),
                                                                                                 u=(u - np.array(
                                                                                                     [0, 0, 0,
                                                                                                      self.m * self.g]) / np.array(
                                                                                                     [self.action_limit,
                                                                                                      self.action_limit,
                                                                                                      self.action_limit,
                                                                                                      self.action_limit * 5 / 2])),
                                                                                                 optimal_x=self.optimal_x,
                                                                                                 max_x_dist=self.max_x_dist,
                                                                                                 min_x_dist=self.min_x_dist,
                                                                                                 alpha=self.alpha,
                                                                                                 beta=self.beta,
                                                                                                 gamma=self.gamma,
                                                                                                 segma=self.segma,
                                                                                                 max_steps=self.max_episode_steps,
                                                                                                 collision=False,
                                                                                                 dt = self.dt
                                                                                                 )
        if self.WandB :
            wandb.log({'x relative ': x})
            wandb.log({'y relative ': y})
            wandb.log({'z relative ': z})

            wandb.log({'vx relative': v_x})
            wandb.log({'vy relative': v_y})
            wandb.log({'vz relative': v_z})

            wandb.log({'ax relative': a_x})
            wandb.log({'ay relative': a_y})
            wandb.log({'az relative': a_z})

            wandb.log({'Target x WRT real world frame ': p_target[0]})
            wandb.log({'Target y WRT real world frame ': p_target[1]})
            wandb.log({'Target z WRT real world frame ': p_target[2]})

            wandb.log({'Tracker x WRT real world frame ': p_tracker[0]})
            wandb.log({'Tracker y WRT real world frame ': p_tracker[1]})
            wandb.log({'Tracker z WRT real world frame ': p_tracker[2]})

            wandb.log({'reward': self.reward})
            wandb.log({'reward_track': reward_track})

            wandb.log({'Depth error': x_error})
            wandb.log({'y in image frame error': y_error})
            wandb.log({'z in image frame error': z_error})

        # Update loop states
        self.Xin = Xout
        self.Tin = Tout

        if self.episode_steps >= self.max_episode_steps:
            self.done = True
        #self.done = False
        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.UE.reset()
        # env and obs VARIABLES
        self.episode_steps = 0
        self.stack_obs = None
        self.feat_vector_history = None

        # INITIAL CONDITIONS
        # Tracker
        pin = np.zeros((3, 1))
        vin = np.array([0, 0, 0]).reshape(3, 1)  # Tracker Initial Velocity[m]
        Rin = np.eye(3)  # Tracker Initial Attitude(Body->Inertial)
        self.fk = self.g * self.m
        self.Xin = np.concatenate((pin, vin, Rin.reshape(-1, 1)), axis=0)

        self.Tin = 0
        # Target
        self.reset_target(pin.reshape((3,)), rand=False)

        # RENDERING
        self.tracker_position_x = []
        self.tracker_position_y = []
        self.tracker_position_z = []
        self.target_position_x = []
        self.target_position_y = []
        self.target_position_z = []

        u = np.zeros((self.action_dim), dtype=np.float64)

        self.step(u)
        self.done = False

        return self.state

    def render(self, mode='human', critic=0, batch_size=400):
        if mode == 'human' :
            if (self.episode_steps % int(self.max_episode_steps/3) == 0):
                self.fig = plt.figure()
                self.ax = plt.axes(projection='3d')
                self.ax.cla()
                self.ax.plot3D(self.tracker_position_x, self.tracker_position_y, self.tracker_position_z, 'b',
                               label='Tracker Trajectory')
                self.ax.plot3D(self.target_position_x, self.target_position_y, self.target_position_z, 'orange',
                               label='Target Trajectory')
                self.ax.legend()
                # Target starting and ending point :
                # self.ax.scatter3D(self.target_position_x[799], self.target_position_y[799], self.target_position_z[799], c='orange', marker='^')
                self.ax.scatter3D(self.target_position_x[0], self.target_position_y[0], self.target_position_z[0],
                                  c='orange', marker='^')
                # Tracker ending and starting point ( green : start and red : end )
                self.ax.scatter3D(self.tracker_position_x[0], self.tracker_position_y[0], self.tracker_position_z[0],
                                  c='green', marker='o')
                # self.ax.scatter3D(self.tracker_position_x[799], self.tracker_position_y[799], self.tracker_position_z[799], c='red', marker='o')

                plt.pause(0.01)
                # plt.show()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
