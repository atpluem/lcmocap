import numpy as np
import time

from numpy.random import randint, rand
from utils.utilfuncs import *

def get_pose_ga_rot(body_segm, df, poses, bpy):
    start = time.time()
    genetic_algo(body_segm['spine'], body_segm['spine']+body_segm['left_arm']+body_segm['right_arm'], df, poses, bpy)
    genetic_algo(body_segm['left_arm'], body_segm['left_arm'], df, poses, bpy)
    genetic_algo(body_segm['right_arm'], body_segm['right_arm'], df, poses, bpy)
    genetic_algo(body_segm['left_leg'], body_segm['left_leg'], df, poses, bpy)
    genetic_algo(body_segm['right_leg'], body_segm['right_leg'], df, poses, bpy)
    stop = time.time()
    print('Take time:', stop-start)

def genetic_algo(body_parts, update_parts, df, poses, bpy):
    Root = ['pelvis', 'spine1', 'left_hip', 'right_hip', 
            'left_collar', 'right_collar']
    
    for part in body_parts:
        if part in Root: continue
        print(body_parts[body_parts.index(part)-1])
        
        pose_bound = [[-1.6, 1.6], [-1.6, 1.6], [-1.6, 1.6]]
        n_iter = 30
        n_bits = 32
        n_pop = 40
        r_cross = 0.9
        r_mut = 1.0 / (float(n_bits) * len(pose_bound))
        parent = body_parts[body_parts.index(part)-1]
        child = part

        # initial population of random bitstring
        pop = [randint(0, 2, n_bits*len(pose_bound)).tolist() for _ in range(n_pop)]
        # track of best solution
        best, best_eval = 0, objective_loss(decode(pose_bound, n_bits, pop[0]),
                                              child, parent, update_parts, df, bpy)
        # enumerate generations
        for gen in range(n_iter):
            # decode population
            decoded = [decode(pose_bound, n_bits, p) for p in pop]
            # evaluate all candidates in the population
            scores = [objective_loss(d, child, parent, update_parts, df, bpy) \
                                    for d in decoded]
            # check for new best solution
            for i in range(n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    print(">%d, new best f(%s) = %f" %(gen,  decoded[i], scores[i]))
            # select parents
            selected = [selection(pop, scores) for _ in range(n_pop)]
            # create the next generation
            children = list()
            for i in range(0, n_pop, 2):
                # selected parents in pairs
                p1, p2 = selected[i], selected[i+1]
                # crossover and mutation
                for c in crossover(p1, p2, r_cross):
                    mutation(c, r_mut)
                    children.append(c)
            pop = children
        
        dec = decode(pose_bound, n_bits, best)
        poses[body_parts[body_parts.index(part)-1]] = \
            bpy.data.objects['DEST'].pose.bones[body_parts[body_parts.index(part)-1]].rotation_quaternion

def objective_loss(pose, child, parent, update_parts, df, bpy):
    # set the pose according to pose parameter
    set_pose(bpy.data.objects['DEST'], parent, pose)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # update new position of each part of body part
    for part in update_parts:              
        df.loc[df['joint'] == part, ['dest_x', 'dest_y', 'dest_z']] = \
            np.array(bpy.data.objects['DEST'].pose.bones[part].head)
    part_df = df.loc[df['joint'] == child]
    parent_df = df.loc[df['joint'] == parent]

    # calculate angle loss
    src_angle = get_2D_angle(part_df, parent_df, 'src')
    dest_angle = get_2D_angle(part_df, parent_df, 'dest')
    loss_ang = abs(src_angle - dest_angle)
    loss = sum(loss_ang)

    if parent_df['joint'].values == 'spine3':
        if df.loc[df['joint'] == 'left_collar']['dest_x'].values < \
           df.loc[df['joint'] == 'right_collar']['dest_x'].values:
           return 10*loss

    if ((parent_df['src_y'].values > part_df['src_y'].values) and \
       (parent_df['dest_y'].values < part_df['dest_y'].values)) or \
       ((parent_df['src_y'].values < part_df['src_y'].values) and \
       (parent_df['dest_y'].values > part_df['dest_y'].values)) or \
       ((parent_df['src_z'].values > part_df['src_z'].values) and \
       (parent_df['dest_z'].values < part_df['dest_z'].values)) or \
       ((parent_df['src_z'].values < part_df['src_z'].values) and \
       (parent_df['dest_z'].values > part_df['dest_z'].values)) or \
       ((parent_df['src_x'].values > part_df['src_x'].values) and \
       (parent_df['dest_x'].values < part_df['dest_x'].values)) or \
       ((parent_df['src_x'].values < part_df['src_x'].values) and \
       (parent_df['dest_x'].values > part_df['dest_x'].values)):
        return 10*loss

    return loss

def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		decoded.append(value)
	return decoded