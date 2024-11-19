import os
import time
from collections import deque

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import airsim
import wandb
import cv2
import imgaug.augmenters as iaa
from scipy.interpolate import CubicSpline
import utilities
import torch
from event.event_sim import EventSimulator
from event.event_processor import EventProcessor
class UnrealEnv():

    def __init__(self,client ,
                 rank,
                 optimal_x=3,
                 dt=0.5,
                 max_episode_steps=400,
                 screen_height=800,screen_width=800,
                 random_pose_init=True,
                 stack_length = 3,
                 vect_size = 8,
                 obs_type = '',
                 debug = True) :
        # AirSim

        self.client = client
        self.rank = rank
        self.Target = f"Target_{self.rank}"
        self.Tracker = f"Tracker_{self.rank}"
        print("rank :", rank , " Target :", self.Target , " Tracker :", self.Tracker )

        #Variabes
        self.Pos_rel_historic = None
        self.Vel_rel_historic = None
        self.random_pose_init = random_pose_init

        # observation shape :
        self.obs_type = obs_type
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.stack_length = stack_length
        self.vect_size = vect_size
        self.z = None
        # Get target and tracker initial positions
        self.optimal_x = optimal_x
        self.dt = dt
        self.max_episode_steps = max_episode_steps

        # event --> latent vector
        if self.obs_type == 'Event_rep_vector'  :
            self.start_ts = 0
            self.init = True
            self.pretrained_weights = r"C:\Users\c23liard\Desktop\Ala_Workspace\Projects\AirSim_codes\myscripts\event_encoder_train\weights\evae_uav_empty_all_vues.pt"
            self.debug = debug
            self.eventsim = EventSimulator(screen_height, screen_width)
            self.eventproc = EventProcessor(
                data_len=3,
                image_size=(screen_height, screen_width, 3),
                pretrained_weights=self.pretrained_weights,
                ls=self.vect_size,
                tc = True,  #True if temporal coding is included
                debug = self.debug,  #Setting this to True will visualize input event stream and reconstructed image
                noise_level=0.0 , # random_table_gen event firing without activity
                sparsity= 0.0 # sparsity (events not firing although there is activity
            )
        elif self.obs_type == 'event_img' :
            self.start_ts = 0
            self.init = True
            self.eventsim = EventSimulator(screen_height, screen_width)

        # Initialise RGB camera
        self.RGB_image_type = airsim.ImageType.Scene
        self.RGB_camera_name = "0"
        responses = self.client.simGetImages(
            [airsim.ImageRequest(self.RGB_camera_name, self.RGB_image_type, False, False)],vehicle_name=self.Tracker)
        response = responses[0]
        self.camera_height, self.camera_width = response.height, response.width
        self.client.simSetDetectionFilterRadius(self.RGB_camera_name, self.RGB_image_type, 80 * 100)  # in [cm]

        # spawn the boxe :
        self.spawn_box()
        self.reset()






    ########################################################################## take off  and move drones
    def hover_drones(self):
        self.client.hoverAsync(self.Tracker)
        self.client.hoverAsync(self.Tracker)

    def take_off_drones(self):
        # Init the drones :
        self.target_takeoff = False
        self.tracker_takeoff = False
        self.client.armDisarm(True, self.Target)
        self.client.armDisarm(True, self.Tracker)
        time.sleep(0.1)
        self.client.takeoffAsync(10, self.Tracker)
        self.client.takeoffAsync(10, self.Target).join()

        #after hovering we set the positions to the initial positions
        pose_target = self.client.simGetVehiclePose(self.Target)
        pose_traker = self.client.simGetVehiclePose(self.Tracker)

        self.pose_target_init =pose_target
        self.pose_tracker_init= pose_traker
        self.pose_target_init.position.x_val = 0
        self.pose_target_init.position.y_val = 0
        self.pose_target_init.position.z_val = 0

        self.pose_tracker_init.position.x_val = 0
        self.pose_tracker_init.position.y_val = 0
        self.pose_tracker_init.position.z_val = 0


        # if the drone didn't take off
        T_z = self.client.getMultirotorState(self.Target).kinematics_estimated.position.z_val
        target_takeoff =not self.client.getMultirotorState(self.Target).landed_state
        time_takeoff = 0
        while not target_takeoff :
            time_takeoff += 1
            if time_takeoff < 15 :
                self.client.armDisarm(False, self.Target)
                self.client.armDisarm(True, self.Target)
            print("target taking off try number ", time_takeoff)
            self.client.takeoffAsync(20,self.Target).join()
            self.client.moveToZAsync(0, 10, 3e+38, {'is_rate': True, 'yaw_or_rate': 0.0}, lookahead=-1,
                                     adaptive_lookahead=1, vehicle_name=self.Target).join()
            T_z = self.client.getMultirotorState(self.Target).kinematics_estimated.position.z_val
            target_takeoff = T_z > -0.2 or T_z < 0.2
            time_takeoff = 0
        Tr_z = self.client.getMultirotorState(self.Tracker).kinematics_estimated.position.z_val
        traker_takeoff = Tr_z > -0.2 or Tr_z < 0.2
        while not traker_takeoff and time_takeoff < 15:
            time_takeoff += 1
            if time_takeoff < 15 :
                self.client.armDisarm(False, self.Tracker)
                self.client.armDisarm(True, self.Tracker)
            print("tracker taking off try number ", time_takeoff)
            self.client.takeoffAsync(20, self.Tracker).join()
            self.client.moveToZAsync(0, 10, 3e+38, {'is_rate': True, 'yaw_or_rate': 0.0}, lookahead=-1,
                                     adaptive_lookahead=1, vehicle_name=self.Tracker).join()
            Tr_z = self.client.getMultirotorState(self.Tracker).kinematics_estimated.position.z_val
            traker_takeoff = Tr_z > -0.2 or Tr_z < 0.2

    # TODO define the offset ( limits are a subspace inside the total workspace )
    def move_target_by_velocities(self,episode_steps):
        x_limit = (-1000000, 1000000)
        y_limit = (-1000000, 1000000)
        z_limit = (-1000000, 0)

        v= self.Target_velocities[episode_steps]

        pose = self.client.simGetVehiclePose(self.Target)
        position = pose.position
        new_x = position.x_val + v[0] * self.dt
        new_y = position.y_val + v[1] * self.dt
        new_z = position.z_val + v[2] * self.dt
        # print('old p', position , 'new p' , new_x, new_y, new_z)
        if new_x < x_limit[0] or new_x > x_limit[1]: v[0] = 0
        if new_y < y_limit[0] or new_y > y_limit[1]: v[1] = 0
        if new_z < z_limit[0] or new_z > z_limit[1]:   v[2] = 0

        # Move the drone
        self.client.moveByVelocityAsync(vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
                                   duration=5, drivetrain=0, vehicle_name=self.Target)
    def move_target_by_pose(self,episode_steps):

        x_limit = (-100000, 100000)
        y_limit = (-100000, 100000)
        z_limit = (-100000, 0)

        v= self.Target_velocities[episode_steps]

        pose = self.client.simGetVehiclePose(self.Target)
        position = pose.position
        new_x = position.x_val + v[0] * self.dt
        new_y = position.y_val + v[1] * self.dt
        new_z = position.z_val + v[2] * self.dt

        # print('old p', position , 'new p' , new_x, new_y, nestions[episode_steps][2]w_z)
        if new_x < x_limit[0] or new_x > x_limit[1]: new_x = position.x_val
        if new_y < y_limit[0] or new_y > y_limit[1]: new_y = position.y_val
        if new_z < z_limit[0] or new_z > z_limit[1]: new_z = position.z_val

        pose.position.x_val = new_x
        pose.position.y_val = new_y
        pose.position.z_val = new_z
        #  [position[0] * 100 + self.offset[0], position[1] *100 + self.offset[1],position[2] * 100 + self.offset[2]]

        self.client.simSetVehiclePose(pose=pose, ignore_collision=True, vehicle_name=self.Target)

    def move_tracker(self,v,yaw_rate):

        pose = self.client.simGetVehiclePose(self.Tracker)
        #Update Yaw
        current_yaw = self.Get_Tracker_Yaw_angle_rel_to_init()
        new_yaw = current_yaw + yaw_rate*self.dt
        pose.orientation.w_val = np.cos(new_yaw / 2) # w_val
        pose.orientation.z_val = np.sin(new_yaw / 2)   # z_val

        # Update position
        pose.position.x_val += v[0]*self.dt
        pose.position.y_val += v[1]*self.dt
        pose.position.z_val += v[2]*self.dt

        self.client.simSetVehiclePose(pose=pose, ignore_collision=True, vehicle_name=self.Tracker)

    def set_tracker_pose(self,pose_list):
        # Assume pose_list contains the values [x, y, z, qx, qy, qz, qw]
        pose = self.client.simGetVehiclePose(self.Tracker)
        pose.position.x_val = pose_list[0]
        pose.position.y_val = pose_list[1]
        pose.position.z_val = pose_list[2]

        pose.orientation.x_val = pose_list[3]
        pose.orientation.y_val = pose_list[4]
        pose.orientation.z_val = pose_list[5]
        pose.orientation.w_val = pose_list[6]
        self.client.simSetVehiclePose(pose=pose, ignore_collision=True, vehicle_name=self.Tracker)

    def set_target_pose(self,pose_list):
        # Assume pose_list contains the values [x, y, z, qx, qy, qz, qw]
        pose = self.client.simGetVehiclePose(self.Target)
        pose.position.x_val = pose_list[0]
        pose.position.y_val = pose_list[1]
        pose.position.z_val = pose_list[2]

        pose.orientation.x_val = pose_list[3]
        pose.orientation.y_val = pose_list[4]
        pose.orientation.z_val = pose_list[5]
        pose.orientation.w_val = pose_list[6]
        self.client.simSetVehiclePose(pose=pose, ignore_collision=True, vehicle_name=self.Target)





    ######################################################################################################################## get states

    def Get_Target_Pos_in_image(self):
        self.client.simClearDetectionMeshNames(self.RGB_camera_name, self.RGB_image_type)
        self.client.simAddDetectionFilterMeshName(self.RGB_camera_name, self.RGB_image_type, self.Target)
        self.detection_info = self.client.simGetDetections(self.RGB_camera_name, self.RGB_image_type)
        if self.detection_info != []:
            x_c , y_c = self.get_detection_centre(self.detection_info)
            return [x_c, y_c]
        else :
            return  None

    def get_detection_centre(self, detection_info):  # 2x1
        box2d_info = detection_info[0].box2D
        # Extracting the maximum and minimum coordinates from box2d_info
        max_coords = (box2d_info.max.x_val, box2d_info.max.y_val)
        min_coords = (box2d_info.min.x_val, box2d_info.min.y_val)

        # Calculate the center of the bounding box
        center_x = (max_coords[0] + min_coords[0]) / 2
        center_y = (max_coords[1] + min_coords[1]) / 2
        # normalize
        center_x = 2 * center_x / self.camera_width
        center_y = 1 * center_y / self.camera_height

        return center_x, center_y

    def Get_Target_Pos_rel_to_init(self):
        pos = self.client.simGetVehiclePose(self.Target).position
        return [pos.x_val, pos.y_val, pos.z_val]

    def Get_Tracker_Pos_rel_to_init(self):
        pos = self.client.simGetVehiclePose(self.Tracker).position
        return [pos.x_val, pos.y_val, pos.z_val]



    def Get_Target_State_rel_to_Tracker(self):
        # Retrieve positions of Tracker and Target
        OM = self.client.simGetVehiclePose(self.Tracker).position
        om = self.client.simGetVehiclePose(self.Target).position
        # Retrive tracker rotation matrix
        quaternion = self.client.simGetVehiclePose(self.Tracker).orientation
        # Assuming quaternion is in the format (x, y, z, w)
        quaternion_array = [quaternion.x_val, quaternion.y_val, quaternion.z_val, quaternion.w_val]
        R_tracker = utilities.quaternion_to_rotation_matrix(quaternion_array)

        # Calculate the relative position
        rel_pos_wrt_world = [-OM.x_val + self.optimal_x + om.x_val,
               om.y_val - OM.y_val,
               -OM.z_val + om.z_val]

        # Initialize the deques if they are None
        if self.Pos_rel_historic is None:
            self.Pos_rel_historic = deque(maxlen=2)
        if self.Vel_rel_historic is None:
            self.Vel_rel_historic = deque(maxlen=2)

        # Append the new position and episode step to the position history
        self.Pos_rel_historic.append(rel_pos_wrt_world)

        # Calculate velocity if there is at least one previous position
        if len(self.Pos_rel_historic) > 1:
            prev_pos = self.Pos_rel_historic[-2]
            rel_vel_wrt_world= [((rel_pos_wrt_world[i] - prev_pos[i])/ 10) / self.dt for i in range(3)]  #(m/s)
        else:
            rel_vel_wrt_world = [0.0, 0.0, 0.0]

        # Append the new position, velocity, and episode step to the velocity history
        self.Vel_rel_historic.append(rel_vel_wrt_world)

        # Calculate acceleration if there is at least one previous velocity
        if len(self.Vel_rel_historic) > 1:
            prev_velocity = self.Vel_rel_historic[-2]
            rel_acc_wrt_world = [(rel_vel_wrt_world[i] - prev_velocity[i]) / self.dt for i in range(3)]
        else:
            rel_acc_wrt_world = [0.0, 0.0, 0.0]

        # change the frame from world frame to tracker frame
        rel_pos_wrt_tracker= np.dot(R_tracker.T, rel_pos_wrt_world)
        rel_vel_wrt_tracker =  np.dot(R_tracker.T, rel_vel_wrt_world)
        rel_acc_wrt_tracker =  np.dot(R_tracker.T, rel_acc_wrt_world)

        return {
            'position': rel_pos_wrt_tracker,
            'velocity': rel_vel_wrt_tracker,
            'acceleration': rel_acc_wrt_tracker,
            'Rotation_matrix': R_tracker
        }

    def Get_Target_Pos_rel_to_world(self):
        pos = self.client.simGetVehiclePose(self.Target).position
        return [pos.x_val+self.optimal_x, pos.y_val, pos.z_val]

    def Get_Tracker_Pos_rel_to_world(self):
        pos = self.client.simGetVehiclePose(self.Tracker).position
        return [pos.x_val, pos.y_val, pos.z_val]

    def Get_Tracker_Yaw_angle_rel_to_init(self):
        quaternion = self.client.simGetVehiclePose(self.Tracker).orientation
        q=np.array([quaternion.w_val, quaternion.x_val, quaternion.y_val, quaternion.z_val]) # [w, x, y, z]
        q_norm = np.linalg.norm(q)
        q_normalized = q / q_norm
        yaw = np.arctan2(2.0 * (q_normalized[0] * q_normalized[3] + q_normalized[1] * q_normalized[2]),
                         1.0 - 2.0 * (q_normalized[2] ** 2 + q_normalized[3] ** 2))
        if yaw < -np.pi: yaw += 2.0 * np.pi
        elif yaw > np.pi: yaw -= 2.0 * np.pi

        return yaw





    def generate_curved_velocity_vectors(self,N):
        damping_factor = 0.1
        velocities = np.zeros((N, 3))
        initial_velocity = np.zeros(3)
        velocities[0] = initial_velocity
        t = 0.0
        ax = 0.2 + random.random() *5
        ay = 0.2 + random.random() *5
        az = 0.2 + random.random() *5
        az *= -1

        kx = 1 + random.random()*2
        ky = 1 + random.random()*2
        kz = 1 + random.random()*2

        phix = -np.pi / 2 + random.random() * np.pi / 2
        phiy = -np.pi / 2 + random.random() * np.pi / 2
        phiz = -np.pi / 2 + random.random() * np.pi / 2

        wx = 0.1* ax
        wy = 0.1 * ay
        wz = 0.1 * az




        for i in range(1, N):
            # Update time
            t += self.dt

            # Calculate velocity components using a sine function with perturbed parameters
            velocity_x = np.cos(wx * t + phix) * ax
            velocity_y = np.cos(wy * t + phiy) * ay
            velocity_z = np.cos(wz * t + phiz) * az

            # Update velocity
            velocities[i] = np.array([velocity_x, velocity_y, velocity_z])
            # Apply damping factor to gradually reduce velocity
            velocities[i] *= (1 - damping_factor)
            velocities[i] = np.clip(velocities[i], -5, 5)
        # add random stop
        for j in range(3) :
            start_index = random.randint(j*int(N/3), int((j+1)*N/3) - int((N/3)*0.05) - 1)
            constant_length = random.randint(1, int((N/3)*0.05))
            for i in range(start_index, start_index + constant_length):
                velocities[i] =np.zeros(3)
        return velocities

    def generate_curved_trajectory(self,N):
        positions = np.zeros((N, 3))
        initial_position = np.zeros(3)
        positions[0] = initial_position
        t = 0.0

        self.a1 = 1 + random.random() * 0.5  # 30
        self.a2 = 1 + random.random() * 0.5  # 30
        self.a3 = 1 + random.random() * 1.5  # 3
        self.a3 *= -1

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


        for i in range(1, N):
            # Update time
            t += self.dt

            # Calculate velocity components using a sine function with perturbed parameters
            pos_x = self.a1 * np.sin(self.phi1 + self.ws1 * t) - self.a1 * np.sin(self.phi1) + positions[0][0]
            pos_y = self.a2 * np.sin(self.phi2 + self.ws2 * t) - self.a2 * np.sin(self.phi2) + positions[0][1]
            pos_z = self.a3 * np.sin(self.phi3 + self.ws3 * t) - self.a3 * np.sin(self.phi3) + positions[0][2]

            # Update velocity
            positions[i] = np.array([pos_x, pos_y, pos_z])

        # stop at random index
        for j in range(3) :
            start_index = random.randint(j*int(N/3), int((j+1)*N/3))
            constant_length = random.randint(1, int((N/3)*0.05))
            for i in range(start_index, start_index + constant_length):
                positions[i] = positions[start_index]

        return positions


    ######################################################################################################################## Reset
    def assign_random_unit_quaternions(self,pose):
        # Convert yaw angle (in radians) to quaternion
        yaw =random.uniform(0, 2 * np.pi)
        yaw_quaternion = np.array([
            np.cos(yaw / 2),  # w_val
            0.0,  # x_val
            0.0,  # y_val
            np.sin(yaw / 2)  # z_val
        ])

        # Assign quaternion values to pose's orientation
        pose.orientation.w_val = yaw_quaternion[0]
        pose.orientation.x_val = yaw_quaternion[1]
        pose.orientation.y_val = yaw_quaternion[2]
        pose.orientation.z_val = yaw_quaternion[3]

    def reset_pose_null(self,pose):
        pose.position.x_val = 0
        pose.position.y_val = 0
        pose.position.z_val = 0
        pose.orientation.w_val = 1
        pose.orientation.x_val = 0
        pose.orientation.y_val = 0
        pose.orientation.z_val = 0
    def reset(self):

        #change box texture
        box_name = f'Training_box_{self.rank -1}'
        self.change_texture(box_name)

        #self.client.reset()
        self.client.enableApiControl(True, self.Target )
        self.client.enableApiControl(True,self.Tracker )


        pose_tracker = self.client.simGetVehiclePose(self.Tracker)
        pose_target = self.client.simGetVehiclePose(self.Target)
        # teleport the drone in x and y directions to a random position
        # Position
        if self.random_pose_init == True :
            target_new_x = random.randint(1, 500)
            target_new_y = random.randint(1, 500)
            pose_target.position.x_val =target_new_x
            pose_target.position.y_val =target_new_y
            pose_target.position.z_val = 0
            # Orientation
            self.assign_random_unit_quaternions(pose_target)
            quaternion= pose_target.orientation
            quaternion_array = [quaternion.x_val, quaternion.y_val, quaternion.z_val, quaternion.w_val]
            rotation_mat = utilities.quaternion_to_rotation_matrix(quaternion_array)
            tracker_position = np.array([target_new_x, target_new_y,0]) + np.array([self.optimal_x,0,0]) +  rotation_mat.dot(np.array([-self.optimal_x,0,0]))

            pose_tracker.position.x_val =tracker_position[0]
            pose_tracker.position.y_val =tracker_position[1]
            pose_tracker.position.z_val = 0
            pose_tracker.orientation = pose_target.orientation


        else :
            self.reset_pose_null(pose_tracker)
            self.reset_pose_null(pose_target)

        self.client.simSetVehiclePose(pose_tracker, True, self.Tracker)

        self.client.simSetVehiclePose(pose_target, True, self.Target)

        self.Pos_rel_historic = None
        self.Vel_rel_historic = None
        # generate target velocities  :
        self.Target_velocities = self.generate_curved_velocity_vectors(self.max_episode_steps+1)
        # generate target positions
        #self.Target_posistions = self.generate_curved_trajectory(self.max_episode_steps+1)
    def teleport_drones_to_init(self):
        self.client.simSetVehiclePose(self.pose_target_init, True, self.Target)
        time.sleep(0.01)
        self.client.simSetVehiclePose(self.pose_tracker_init, True, self.Tracker)
        time.sleep(0.01)


    ################################################################################################################## Get images
    def get_gray_image(self):
        # get rgb image
        # scene vision image in png format
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        ],vehicle_name=self.Tracker)
        # check observation
        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            ],vehicle_name=self.Tracker)

        # get gary image
        img_1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        # reshape array to 4 channel image array H X W X 3
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # Resize grayscale image
        resized_gray_image = cv2.resize(img_gray, (self.screen_width, self.screen_height))

        # cv2.imshow('test', resized_gray_image)
        # cv2.waitKey(1)

        return resized_gray_image

    def get_RGB_image(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        ], vehicle_name=self.Tracker)

        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            ], vehicle_name=self.Tracker)

        img_1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)

        # Resize RGB image
        resized_rgb_image = cv2.resize(img_rgb, (self.screen_width, self.screen_height))

        # Convert RGB to BGR
        resized_bgr_image = cv2.cvtColor(resized_rgb_image, cv2.COLOR_RGB2BGR)

        #cv2.imshow('test', resized_rgb_image)
        #cv2.waitKey(1)

        return resized_rgb_image

    def computeZ(self,img, ts):
        with torch.no_grad():
            # cv2.imwrite(f"rgb_{self.idx}.png", img)
            # self.idx += 1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            img = cv2.add(img, 0.001)

            # Receive event image and list of events from simulator
            event_img, n_pix_ev, events = self.eventsim.image_callback(img, ts)

            bytestream = []

            if events is not None and events.shape[0] > 0:
                bytestream = events.tolist()

            # Encode event list into a latent vector

            if len(bytestream) > 0:
                z = (
                    self.eventproc.convertStream(bytestream, ts, n_pix_ev)
                    .cpu()
                    .numpy()
                )
            else:
                z = np.zeros([1, self.latent_vect_size])

        return z

    def get_obs(self):
        if self.obs_type == 'RGB'  :
            # output : (stack_size=3, C=3, H, W)
            # get RGB image (HxWxC)
            img  = self.get_RGB_image()
            img = np.transpose(img, (2, 0, 1))
            # Normalize image
            obs_new = (img.astype(np.float32) / 255)

        elif self.obs_type == 'Gray' :
            #get Gray image
            img  = self.get_gray_image()
            obs_new = np.expand_dims(img, axis=0)

        elif self.obs_type == 'Event_rep_vector' :
            # get RGB image (HxWxC)
            img  = self.get_RGB_image()
            ts = time.time_ns()
            if self.init:
                self.start_ts = ts
                self.init = False
            obs_new = self.computeZ(img, (ts - self.start_ts))

        elif self.obs_type == 'event_img' :
            # output : (stack_size=3, C=1, H, W)
            # get RGB image (HxWxC)
            img  = self.get_RGB_image()
            ts = time.time_ns()
            if self.init:
                self.start_ts = ts
                self.init = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            img = cv2.add(img, 0.001)

            # Receive event image and list of events from simulator
            event_img_bw, n_pix_ev, events = self.eventsim.image_callback(img, ts- self.start_ts)
            bytestream = []

            if events is not None and events.shape[0] > 0:
                bytestream = events.tolist()
            if len(bytestream) > 0:
                event_img_rb = self.convert_event_img_rb(events)
            else:
                event_img_rb = np.zeros((self.screen_height, self.screen_width, 2), np.float32)


            event_img_bw = event_img_bw.reshape([self.screen_height,self.screen_width,1])
            cv2.imshow('test',np.concatenate((event_img_rb,
                                              np.zeros((self.screen_height, self.screen_width, 1),
                                                       np.float32)), axis=-1))
            cv2.waitKey(1)
            # Transpose to correct shape
            event_img_bw = np.transpose(event_img_bw, (2, 0, 1))
            event_img_rb = np.transpose(event_img_rb, (2, 0, 1))

            obs_new = event_img_rb

        return obs_new

    def convert_event_img_rb(self, events):
        n_events_total = len(events)
        start_idx = 0
        t_start = events[start_idx][0]
        idx = start_idx
        frame_events = []

        # Create empty frame with 2 channels (red and blue)
        self.frame = np.zeros((self.screen_height, self.screen_width, 2), np.float32)

        n_events = n_events_total
        t_final = events[n_events - 1][0]
        dt = t_final - t_start
        if dt < 1e-3:
            dt = 1e-3

        # Stack events
        while idx - start_idx < n_events:
            e_curr = events[idx]
            frame_events.append(e_curr)
            # Timestamps are relative to window of time observed in the event data
            t_relative = float(t_final - e_curr[0]) / dt
            frame_events[-1][0] = t_relative
            idx += 1

        idx = 0
        for e in frame_events:  # T, X, Y, P
            x, y = int(e[1]), int(e[2])
            value = e[3] * e[0]
            if value < 0:
                # Blue channel for negative values
                self.frame[x, y, 1] = -value
            elif value > 0:
                # Red channel for positive values
                self.frame[x, y, 0] = value
            idx += 1

        # Clip the frame values to be within [0, 1]
        self.frame = np.clip(self.frame, 0, 1)

        return self.frame

    def visualize_stacked_images(self, stack_obs):
        image1 = stack_obs[0, :, :]  # Extracting channels 0 to 2
        image2 = stack_obs[1, :, :]  # Extracting channels 4 to 6
        image3 = stack_obs[2, :, :]  # Extracting channels 8 to 10

        resized_image1 = cv2.resize(image1, (1000, 600))
        resized_image2 = cv2.resize(image2, (1000, 600))
        resized_image3 = cv2.resize(image3, (1000, 600))

        script_dir = os.path.dirname(__file__)
        images_dir = os.path.join(script_dir, 'stacked_obs_AsymCNN')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        # Save the resized images in the "images" folder
        cv2.imwrite(os.path.join(images_dir, f'stacked_obs_1_{self.episode_steps}.jpg'), resized_image1)
        cv2.imwrite(os.path.join(images_dir, f'stacked_obs_2_{self.episode_steps}.jpg'), resized_image2)
        cv2.imwrite(os.path.join(images_dir, f'stacked_obs_3_{self.episode_steps}.jpg'), resized_image3)

        # Display the images
        # cv2.imshow('First Image', resized_image1)
        # cv2.imshow('Second Image', resized_image2)
        # cv2.imshow('Third Image', resized_image3)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def spawn_box(self):
        box_pose = airsim.Pose(airsim.Vector3r(x_val=0,y_val= 65 * (self.rank - 1),z_val= 5.5),
                               airsim.Quaternionr(x_val=0.0, y_val=0.0, z_val=0, w_val=1.0))
        box_scale = airsim.Vector3r(1, 1, 1)
        box_name = f'Training_box_{self.rank}'
        self.client.simSpawnObject(box_name, f"Training_box_Blueprint_{self.rank}", box_pose, box_scale,is_blueprint= True)
        print("Main box spawned at position",0, 65 * (self.rank - 1), 5.5)
    def change_texture(self,object_name):
        texture_index = random.choice(range(10))
        tag = object_name
        self.client.simSwapTextures(tag, texture_index)

