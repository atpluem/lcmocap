import numpy as np
import time

from tqdm import tqdm
from numpy.random import randint, rand
from utils.utilfuncs import *

def get_pose_ga(body_segm, df, poses, bpy, scales, visualize):
    total_loss = dict()
    
    # initialize visualize
    sc = 0
    if visualize:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        sc = ax.scatter3D(df['dest_x'], df['dest_y'], df['dest_z'])
        fig.show()
    
    # start retargeting    
    start = time.time()
    for body_set in tqdm(body_segm, desc='Retargeting'):
        if body_set == 'spine':
            update_part = body_segm[body_set] + body_segm['left_arm'] + \
                            body_segm['right_arm']
        else: update_part = body_segm[body_set]
        genetic_algo(body_set, body_segm[body_set], update_part, df, 
                        poses, bpy, scales, total_loss, sc, visualize)
    elapsed = time.time() - start
    tqdm.write('Retargeting done after {:.4f} seconds'.format(elapsed))
    tqdm.write('Retargeting final loss val = {:.4f}'.format(get_sum_total_loss(total_loss)))
    return total_loss

def genetic_algo(body_set, body_parts, update_parts, df, poses, bpy, scales, total_loss, sc, visualize):
    Root = ['pelvis', 'spine1', 'left_hip', 'right_hip', 
            'left_collar', 'right_collar']
    
    for part in tqdm(body_parts, desc='Stage {:10s}'.format(body_set)):
        stage_start = time.time()
        if part in Root: continue
        
        pose_bound = [[-1.6, 1.6], [-1.6, 1.6], [-1.6, 1.6]]
        n_iter = 30
        n_bits = 32
        n_pop = 40
        r_cross = 0.9
        r_mut = 1.0 / (float(n_bits) * len(pose_bound))
        parent = body_parts[body_parts.index(part)-1]
        child = part
        loss_list = []

        # initial population of random bitstring
        pop = [randint(0, 2, n_bits*len(pose_bound)).tolist() for _ in range(n_pop)]
        # track of best solution
        best, best_eval = 0, objective_loss(decode(pose_bound, n_bits, pop[0]),
                                              child, parent, update_parts, df, bpy, scales)
        # enumerate generations
        for gen in range(n_iter):
            # decode population
            decoded = [decode(pose_bound, n_bits, p) for p in pop]
            # evaluate all candidates in the population
            scores = [objective_loss(d, child, parent, update_parts, df, bpy, scales) \
                                    for d in decoded]
            # check for new best solution
            for i in range(n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    # print(">%d, new best f(%s) = %f" %(gen,  decoded[i], scores[i]))
                    if visualize:
                        plt.pause(0.0001)
                        sc._offsets3d = (df['dest_x'], df['dest_y'], df['dest_z'])
                        plt.draw()
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

            loss_list.append(best_eval)
        
        dec = decode(pose_bound, n_bits, best)
        poses[body_parts[body_parts.index(part)-1]] = \
            bpy.data.objects['DEST'].pose.bones[body_parts[body_parts.index(part)-1]].rotation_quaternion
        total_loss[part] = loss_list

        # print stage and time usage
        elapsed = time.time() - stage_start
        tqdm.write('--> {:10s} done after {:.4f} seconds'.format(part, elapsed))

def objective_loss(pose, child, parent, update_parts, df, bpy, scales):
    # set the pose according to pose parameter
    set_pose(bpy.data.objects['DEST'], parent, pose)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # update new position of each part of body part
    for part in update_parts:              
        df.loc[df['joint'] == part, ['dest_x', 'dest_y', 'dest_z']] = \
            np.array(bpy.data.objects['DEST'].pose.bones[part].head)
    part_df = df.loc[df['joint'] == child]
    parent_df = df.loc[df['joint'] == parent]

    # calculate loss expected position of destinstion
    expect_x = part_df['src_x'].values*scales['dest_scale'][0]/scales['src_scale'][0]
    observe_x = part_df['dest_x'].values
    expect_y = part_df['src_y'].values*scales['dest_scale'][1]/scales['src_scale'][1]
    observe_y = part_df['dest_y'].values
    expect_z = part_df['src_z'].values*scales['dest_scale'][2]/scales['src_scale'][2]
    observe_z = part_df['dest_z'].values
    x = abs(expect_x-observe_x)
    y = abs(expect_y-observe_y)
    z = abs(expect_z-observe_z)

    # total loss
    loss = x+y+z

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

    return loss[0]

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