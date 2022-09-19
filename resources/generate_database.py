import quat
import bvh
from scipy.interpolate import griddata
import scipy.signal as signal
import scipy.ndimage as ndimage
import struct
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

""" Files to Process """

files = [
    'run1_subject5.bvh',
    'walk1_subject5.bvh',
    'dance2_subject5.bvh',
    'fight1_subject5.bvh',
]

""" We will accumulate data in these lists """

bone_positions = []
bone_rotations = []
bone_parents = []
bone_names = []
    
range_starts = []
range_stops = []

""" Loop Over Files """

for filename in files:

    """ Load Data """
    
    print('Loading "%s"...' % filename)
    
    bvh_data = bvh.load(filename)
    bvh_data['positions'] = bvh_data['positions']
    bvh_data['rotations'] = bvh_data['rotations']
    
    positions = bvh_data['positions']
    rotations = quat.unroll(quat.from_euler(np.radians(bvh_data['rotations']), order=bvh_data['order']))

    # Convert from cm to m
    positions *= 0.01
    
    """ Supersample """
    
    nframes = positions.shape[0]
    nbones = positions.shape[1]
    
    # Supersample data to 60 fps
    original_times = np.linspace(0, nframes - 1, nframes)
    sample_times = np.linspace(0, nframes - 1, int(0.9 * (nframes * 2 - 1))) # Speed up data by 10%
    
    # This does a cubic interpolation of the data for supersampling and also speeding up by 10%
    positions = griddata(original_times, positions.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), nbones, 3])
    rotations = griddata(original_times, rotations.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), nbones, 4])
    
    # Need to re-normalize after super-sampling
    rotations = quat.normalize(rotations)
    
    """ Extract Simulation Bone """
    
    # First compute world space positions/rotations
    global_rotations, global_positions = quat.fk(rotations, positions, bvh_data['parents'])
    
    # Specify joints to use for simulation bone 
    sim_position_joint = bvh_data['names'].index("Spine2")
    sim_rotation_joint = bvh_data['names'].index("Hips")
    
    # Position comes from spine joint
    sim_position = np.array([1.0, 0.0, 1.0]) * global_positions[:,sim_position_joint:sim_position_joint+1]
    sim_position = signal.savgol_filter(sim_position, 31, 3, axis=0, mode='interp')
    
    # Direction comes from projected hip forward direction
    sim_direction = np.array([1.0, 0.0, 1.0]) * quat.mul_vec(global_rotations[:,sim_rotation_joint:sim_rotation_joint+1], np.array([0.0, 1.0, 0.0]))

    # We need to re-normalize the direction after both projection and smoothing
    sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[...,np.newaxis]
    sim_direction = signal.savgol_filter(sim_direction, 61, 3, axis=0, mode='interp')
    sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1)[...,np.newaxis])
    
    # Extract rotation from direction
    sim_rotation = quat.normalize(quat.between(np.array([0, 0, 1]), sim_direction))

    # Transform first joints to be local to sim and append sim as root bone
    positions[:,0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:,0:1] - sim_position)
    rotations[:,0:1] = quat.mul(quat.inv(sim_rotation), rotations[:,0:1])
    
    positions = np.concatenate([sim_position, positions], axis=1)
    rotations = np.concatenate([sim_rotation, rotations], axis=1)
    rotations = quat.unroll(rotations)
    
    bone_parents = np.concatenate([[-1], bvh_data['parents'] + 1])
    
    bone_names = ['Simulation'] + bvh_data['names']
    
    """ Compute Velocities """
    
    # Compute velocities via central difference
    velocities = np.empty_like(positions)
    velocities[1:-1] = (
        0.5 * (positions[2:  ] - positions[1:-1]) * 60.0 +
        0.5 * (positions[1:-1] - positions[ :-2]) * 60.0)
    velocities[ 0] = velocities[ 1] - (velocities[ 3] - velocities[ 2])
    velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])
    
    # Same for angular velocities
    angular_velocities = np.zeros_like(positions)
    angular_velocities[1:-1] = (
        0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:  ], rotations[1:-1]))) * 60.0 +
        0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[ :-2]))) * 60.0)
    angular_velocities[ 0] = angular_velocities[ 1] - (angular_velocities[ 3] - angular_velocities[ 2])
    angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

    """ Compute Contact Data """ 

    global_rotations, global_positions, global_velocities, global_angular_velocities = quat.fk_vel(
        rotations, 
        positions, 
        velocities,
        angular_velocities,
        bone_parents)
    
    contact_velocity_threshold = 0.15
    
    contact_velocity = np.sqrt(np.sum(global_velocities[:,np.array([
        bone_names.index("LeftToe"), 
        bone_names.index("RightToe")])]**2, axis=-1))
    
    # Contacts are given for when contact bones are below velocity threshold
    contacts = contact_velocity < contact_velocity_threshold
    
    # Median filter here acts as a kind of "majority vote", and removes
    # small regions  where contact is either active or inactive
    for ci in range(contacts.shape[1]):
    
        contacts[:,ci] = ndimage.median_filter(
            contacts[:,ci], 
            size=6, 
            mode='nearest')
    
    """ Append to Database """
    
    bone_positions.append(positions)
    bone_rotations.append(rotations)
    
    offset = 0 if len(range_starts) == 0 else range_stops[-1] 

    range_starts.append(offset)
    range_stops.append(offset + len(positions))
    
    
""" Concatenate Data """
    
bone_positions = np.concatenate(bone_positions, axis=0).astype(np.float32)
bone_rotations = np.concatenate(bone_rotations, axis=0).astype(np.float32)
bone_parents = bone_parents.astype(np.int32)

range_starts = np.array(range_starts).astype(np.int32)
range_stops = np.array(range_stops).astype(np.int32)

""" Visualize """

def halflife_to_damping(halflife, eps=1e-5):
    return (4.0 * 0.69314718056) / (halflife + eps)


def decay(
    x,        # Initial Position
    v,        # Initial Velocity
    halflife, # Halflife
    dt        # Time Delta
    ):
    
    y = halflife_to_damping(halflife) / 2.0

    return np.exp(-y*dt)*(x + (v + x*y)*dt)
    

def decay_velocity(
    v,        # Initial Velocity
    halflife, # Halflife
    dt        # Time Delta
    ):
    
    y = halflife_to_damping(halflife) / 2.0

    return np.exp(-y*dt)*v*dt

def decay_cubic(
    x, 
    v, 
    blend_time, 
    dt):

    t = np.clip(dt / blend_time, 0, 1)

    d = x
    c = v * blend_time
    b = -3*d - 2*c
    a = 2*d + c
    
    return a*t*t*t + b*t*t + c*t + d
    
    
def decay_velocity_cubic(
    v, 
    blend_time, 
    dt):

    t = np.clip(dt / blend_time, 0, 1)

    c = v * blend_time
    b = -2*c
    a = c
    
    return a*t*t*t + b*t*t + c*t

def softfade(x, hardness=1.0):
    return np.log(1 + np.exp(hardness - 2*hardness*x)) / hardness

def softfade_grad(x, hardness=1.0):
    return (-2*np.exp(hardness)) / (np.exp(2*hardness*x) + np.exp(hardness))

def softfade_initial_grad(hardness=1.0):
    return (-2*np.exp(hardness)) / (1 + np.exp(hardness))

def decay_softfade(
    y_scale,
    x_scale,
    hardness,
    time):
    
    return y_scale * softfade(time/x_scale, hardness)

def decay_softfade_grad(
    y_scale,
    x_scale,
    hardness,
    time):
    
    return y_scale * (softfade_grad(time/x_scale, hardness) / x_scale)

def decay_softfade_initial_grad(
    y_scale,
    x_scale,
    hardness):
    
    return y_scale * (softfade_initial_grad(hardness) / x_scale)




visualize_looper = True

if visualize_looper:
    
    start, stop = 500+162+66, 500+162+367
    bone, component = 0, 0
    
    part = bone_positions[start:stop,bone,component].copy()
    part_velocity = abs(np.gradient(part))
    part_adjust = np.cumsum(part_velocity) / np.sum(part_velocity)
    
    offset = part[-1] - part[0]
    
    linear_offset = np.linspace(offset/2, -offset/2, len(part))
    adjust_offset = -offset * part_adjust + offset/2
    
    velocity_start = part[1] - part[0]
    velocity_end = part[-1] - part[-2]
    velocity_offset = velocity_end - velocity_start
    
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.4))
    fig.suptitle('Unlooped')
    ax.plot(np.arange(len(part)), part, color='red')
    ax.plot(len(part) - 1 + np.arange(len(part)), part, color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Displacement')
    ax.axes.xaxis.set_ticklabels([])
    ax.set_ylim([-3, 3])
    plt.tight_layout()
    plt.show()
    
    start_spring_offset = decay(offset, velocity_offset, 20.0, np.arange(len(part)))
    end_spring_offset = decay(-offset, velocity_offset, 20.0, np.arange(len(part))[::-1])
    
    start_spring_half_offset = decay(offset/2, velocity_offset/2, 20.0, np.arange(len(part)))
    end_spring_half_offset = decay(-offset/2, velocity_offset/2, 40.0, np.arange(len(part))[::-1])
    final_spring_offset = start_spring_half_offset + end_spring_half_offset
    
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.8))
    fig.suptitle('Inertialize Start')
    ax[0].plot(np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(np.arange(len(part)), part + start_spring_offset, color='red')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part + start_spring_offset, color='red')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Displacement')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[0].set_ylim([-3, 3])
    ax[1].plot(np.arange(len(part)), start_spring_offset, color='green', linestyle='dashed')
    ax[1].plot(len(part) - 1 + np.arange(len(part)), start_spring_offset, color='green', linestyle='dashed')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Offset')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[1].set_ylim([-3, 3])
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.8))
    fig.suptitle('Inertialize End')
    ax[0].plot(np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(np.arange(len(part)), part + end_spring_offset, color='red')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part + end_spring_offset, color='red')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Displacement')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[0].set_ylim([-3, 3])
    ax[1].plot(np.arange(len(part)), end_spring_offset, color='green', linestyle='dashed')
    ax[1].plot(len(part) - 1 + np.arange(len(part)), end_spring_offset, color='green', linestyle='dashed')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Offset')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[1].set_ylim([-3, 3])
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.8))
    fig.suptitle('Inertialize Both')
    ax[0].plot(np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(np.arange(len(part)), part + final_spring_offset, color='red')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part + final_spring_offset, color='red')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Displacement')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[0].set_ylim([-3, 3])
    ax[1].plot(np.arange(len(part)), final_spring_offset, color='green', linestyle='dashed')
    ax[1].plot(len(part) - 1 + np.arange(len(part)), final_spring_offset, color='green', linestyle='dashed')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Offset')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[1].set_ylim([-3, 3])
    plt.tight_layout()
    plt.show()
    
    start_cubic_half_offset = decay_cubic(offset/2, velocity_offset/2, 100.0, np.arange(len(part)))
    end_cubic_half_offset = decay_cubic(-offset/2, velocity_offset/2, 100.0, np.arange(len(part))[::-1])
    final_cubic_offset = start_cubic_half_offset + end_cubic_half_offset
    
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.8))
    fig.suptitle('Inertialize Cubic')
    ax[0].plot(np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(np.arange(len(part)), part + final_cubic_offset, color='red')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part + final_cubic_offset, color='red')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Displacement')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[0].set_ylim([-3, 3])
    ax[1].plot(np.arange(len(part)), final_cubic_offset, color='green', linestyle='dashed')
    ax[1].plot(len(part) - 1 + np.arange(len(part)), final_cubic_offset, color='green', linestyle='dashed')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Offset')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[1].set_ylim([-3, 3])
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.8))
    fig.suptitle('Linear Offset')
    ax[0].plot(np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(np.arange(len(part)), part + linear_offset, color='red')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part + linear_offset, color='red')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Displacement')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[0].set_ylim([-3, 3])
    ax[1].plot(np.arange(len(part)), linear_offset, color='green', linestyle='dashed')
    ax[1].plot(len(part) - 1 + np.arange(len(part)), linear_offset, color='green', linestyle='dashed')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Offset')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[1].set_ylim([-3, 3])
    plt.tight_layout()
    plt.show()
    
    start_offset = decay_velocity_cubic(velocity_offset / 2, 50.0, np.arange(len(part)))
    end_offset   = decay_velocity_cubic(velocity_offset / 2, 50.0, np.arange(len(part))[::-1])
    final_offset = linear_offset + start_offset + end_offset
    
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.4))
    fig.suptitle('Velocity Offset')
    ax.plot(start_offset + end_offset, color='green', linestyle='dashed')
    ax.set_xlabel('')
    ax.set_ylabel('Offset')
    ax.axes.xaxis.set_ticklabels([])
    ax.set_ylim([-3, 3])
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.8))
    fig.suptitle('Linear Offset + Velocity Offset')
    ax[0].plot(np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(np.arange(len(part)), part + final_offset, color='red')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part + final_offset, color='red')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Displacement')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[0].set_ylim([-3, 3])
    ax[1].plot(np.arange(len(part)), final_offset, color='green', linestyle='dashed')
    ax[1].plot(len(part) - 1 + np.arange(len(part)), final_offset, color='green', linestyle='dashed')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Offset')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[1].set_ylim([-3, 3])
    plt.tight_layout()
    plt.show()
    
    # TODO: Different widths on each side
    
    start_linear_offset = decay_softfade(offset/2, 75.0, 10.0, np.arange(len(part)))
    end_linear_offset = decay_softfade(-offset/2, 150.0, 10.0, np.arange(len(part))[::-1])
    
    velocity_fade_start = decay_softfade_initial_grad(offset/2, 75.0, 10.0)
    velocity_fade_end = decay_softfade_initial_grad(offset/2, 150.0, 10.0)
    velocity_fade_offset = (velocity_end + velocity_fade_end) - (velocity_start + velocity_fade_start)
    
    fade_start_offset = decay_velocity_cubic(velocity_fade_offset / 2, 50.0, np.arange(len(part)))
    fade_end_offset   = decay_velocity_cubic(velocity_fade_offset / 2, 50.0, np.arange(len(part))[::-1])
    
    fade_offset = start_linear_offset + end_linear_offset + fade_start_offset + fade_end_offset

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    fig.suptitle('$\\frac{\\log(1 + \\exp(\\alpha - 2\\ \\alpha\\ x))}{\\alpha}$')
    ax.plot(np.linspace(0, 1, 100), softfade(np.linspace(0, 1, 100), 5.0), label='$\\alpha=5$')
    ax.plot(np.linspace(0, 1, 100), softfade(np.linspace(0, 1, 100), 10.0), label='$\\alpha=10$')
    ax.plot(np.linspace(0, 1, 100), softfade(np.linspace(0, 1, 100), 25.0), label='$\\alpha=25$')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 2.4))
    fig.suptitle('Softfade Offset')
    ax.plot(start_linear_offset + end_linear_offset, color='green', linestyle='dashed')
    ax.set_xlabel('')
    ax.set_ylabel('Offset')
    ax.axes.xaxis.set_ticklabels([])
    ax.set_ylim([-3, 3])
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.8))
    fig.suptitle('Softfade Offset + Velocity Offset')
    ax[0].plot(np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part, color='red', alpha=0.25, linestyle='dotted')
    ax[0].plot(np.arange(len(part)), part + fade_offset, color='red')
    ax[0].plot(len(part) - 1 + np.arange(len(part)), part + fade_offset, color='red')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Displacement')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[0].set_ylim([-3, 3])
    ax[1].plot(np.arange(len(part)), fade_offset, color='green', linestyle='dashed')
    ax[1].plot(len(part) - 1 + np.arange(len(part)), fade_offset, color='green', linestyle='dashed')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Offset')
    ax[1].axes.xaxis.set_ticklabels([])
    ax[1].set_ylim([-3, 3])
    plt.tight_layout()
    plt.show()    

    root_pos_part0 = bone_positions[1560:1583,0].copy()
    root_rot_part0 = bone_rotations[1560:1583,0].copy()
    
    root_pos_part1 = quat.mul_vec(root_rot_part0[-1:], quat.inv_mul_vec(root_rot_part0[:1], root_pos_part0 - root_pos_part0[:1])) + root_pos_part0[-1:]
    
    root_vel_part0 = root_pos_part0[-1] - root_pos_part0[-2]
    root_vel_part1 = quat.mul_vec(root_rot_part0[-1:], quat.inv_mul_vec(root_rot_part0[0], root_pos_part0[1] - root_pos_part0[0]))
    root_vel_offset = root_vel_part0 - root_vel_part1
    
    root_offset0 = decay_velocity_cubic(root_vel_offset / 2, 15.0, np.arange(len(root_pos_part0))[::-1][...,None])    
    root_offset1 = decay_velocity_cubic(root_vel_offset / 2, 15.0, np.arange(len(root_pos_part0))[...,None])
    
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    fig.suptitle('Root Motion')
    ax.plot(root_pos_part0[...,0], root_pos_part0[...,2], color='red', alpha=0.25, linestyle='dotted')
    ax.plot(root_pos_part1[...,0], root_pos_part1[...,2], color='red', alpha=0.25, linestyle='dotted')
    ax.plot(root_offset0[...,0] + root_pos_part0[...,0], root_offset0[...,2] + root_pos_part0[...,2], color='red')
    ax.plot(root_offset1[...,0] + root_pos_part1[...,0], root_offset1[...,2] + root_pos_part1[...,2], color='red')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.set_xlim([-3.75, -2])
    ax.set_ylim([0.5, 1.75])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


""" Write Database """

print("Writing Database...")

with open('database.bin', 'wb') as f:
    
    nframes = bone_positions.shape[0]
    nbones = bone_positions.shape[1]
    nranges = range_starts.shape[0]
    
    f.write(struct.pack('II', nframes, nbones) + bone_positions.ravel().tobytes())
    f.write(struct.pack('II', nframes, nbones) + bone_rotations.ravel().tobytes())
    f.write(struct.pack('I', nbones) + bone_parents.ravel().tobytes())
    
    f.write(struct.pack('I', nranges) + range_starts.ravel().tobytes())
    f.write(struct.pack('I', nranges) + range_stops.ravel().tobytes())

    
    