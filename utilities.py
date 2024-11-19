import numpy as np
import imgaug.augmenters as iaa
import wandb


def quaternion_to_rotation_matrix( quaternion):
    x, y, z, w = quaternion
    rotation_matrix = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
    ])
    return rotation_matrix
def augment_image(self, image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.shape[0] == 3:
        # Transpose the image from (channels, height, width) to (height, width, channels)
        image = image.transpose(1, 2, 0)
    # Define the sequence of augmentations
    augmentation = iaa.Sequential([
        iaa.Multiply((0.8, 1.2)),  # change brightness
        iaa.LinearContrast((0.75, 1.5)),  # change contrast
        iaa.AddToHueAndSaturation((-20, 20))  # change hue and saturation
    ], random_order=True)

    # Apply the augmentation to the image
    augmented_image = augmentation(image=image)
    augmented_image = augmented_image.transpose(2, 0, 1)

    return augmented_image



def rotation_matrix_to_euler_angles(self, R):
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    # Convert radians to degrees
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    euler_angles = [roll, pitch, yaw]
    return euler_angles


def wandb_log(x, y, z, v_x, v_y, v_z,a_x,a_y,a_z, cur_target_pos, cur_tracker_pos,reward,reward_track,x_error,y_error,z_error):
    wandb.log({'x relative ': x})
    wandb.log({'y relative ': y})
    wandb.log({'z relative ': z})

    wandb.log({'vx relative': v_x})
    wandb.log({'vy relative': v_y})
    wandb.log({'vz relative': v_z})

    wandb.log({'ax relative': a_x})
    wandb.log({'ay relative': a_y})
    wandb.log({'az relative': a_z})

    wandb.log({'Target x WRT real world frame ': cur_target_pos[0]})
    wandb.log({'Target y WRT real world frame ': cur_target_pos[1]})
    wandb.log({'Target z WRT real world frame ': cur_target_pos[2]})

    wandb.log({'Tracker x WRT real world frame ': cur_tracker_pos[0]})
    wandb.log({'Tracker y WRT real world frame ': cur_tracker_pos[1]})
    wandb.log({'Tracker z WRT real world frame ': cur_tracker_pos[2]})

    wandb.log({'reward': reward})
    wandb.log({'reward_track': reward_track})

    wandb.log({'Depth error': x_error})
    wandb.log({'y in image frame error': y_error})
    wandb.log({'z in image frame error': z_error})