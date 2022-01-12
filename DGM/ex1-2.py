import argparse
import tensorflow as tf
import json
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import model3 as model

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='/home/lab1501/Cirp2022DGM/DGM/train_results_mult/trained_models_ex1_2/epoch_1000.ckpt', help='Model checkpoint path')
FLAGS = parser.parse_args()
class_name = raw_input('this is the model of ex1_2,please confirm ').lower()



latent1 = np.load( "/home/lab1501/Cirp2022DGM/data/shape_net_core_uniform_samples_2048/04379247/latent1.npy" )
latent2 = np.load( "/home/lab1501/Cirp2022DGM/data/shape_net_core_uniform_samples_2048/04379247/latent2.npy" )
latent3 = np.load( "/home/lab1501/Cirp2022DGM/data/shape_net_core_uniform_samples_2048/04379247/latent3.npy")
latent = latent3+1*(latent2-latent1)

latentu =  latent[np.newaxis,np.newaxis,np.newaxis,:]
# DEFAULT SETTINGS
pretrained_model_path = FLAGS.model_path # os.path.join(BASE_DIR, './pretrained_model/model.ckpt')
hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')
ply_data_dir = os.path.join(BASE_DIR, './PartAnnotation')
gpu_to_use = 0
output_dir = os.path.join(BASE_DIR, './test_resultsimage')
output_verbose = True   # If true, output all color-coded part segmentation obj files

# MAIN SCRIPT
point_num = 2048            # the max number of points in the all testing data shapes
batch_size = 1

test_file_list = os.path.join(BASE_DIR, 'testing_ply_file_list.txt')

# oid2cpid = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))

# object2setofoid = {}
# for idx in range(len(oid2cpid)):
#     objid, pid = oid2cpid[idx]
#     if not objid in object2setofoid.keys():
#         object2setofoid[objid] = []
#     object2setofoid[objid].append(idx)

# all_obj_cat_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
# fin = open(all_obj_cat_file, 'r')
# lines = [line.rstrip() for line in fin.readlines()]
# objcats = [line.split()[1] for line in lines]
# objnames = [line.split()[0] for line in lines]
# on2oid = {objcats[i]:i for i in range(len(objcats))}
# fin.close()

# color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
# color_map = json.load(open(color_map_file, 'r'))

NUM_OBJ_CATS = 16
NUM_PART_CATS = 3

# cpid2oid = json.load(open(os.path.join(hdf5_data_dir, 'catid_partid_to_overallid.json'), 'r'))

def printout(flog, data):
	print(data)
	flog.write(data + '\n')

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]

            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def placeholder_inputs():
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))
    input_label_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))
    return pointclouds_ph, input_label_ph

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def load_pts_seg_files(pts_file, seg_file, catid):
    with open(pts_file, 'r') as f:
        pts_str = [item.rstrip() for item in f.readlines()]
        pts = np.array([np.float32(s.split()) for s in pts_str], dtype=np.float32)
    with open(seg_file, 'r') as f:
        part_ids = np.array([int(item.rstrip()) for item in f.readlines()], dtype=np.uint8)
        seg = np.array([cpid2oid[catid+'_'+str(x)] for x in part_ids])
    return pts, seg

def pc_augment_to_point_num(pts, pn):
    # assert(pts.shape[0] <= pn)
    cur_len = pts.shape[0]
    res = np.array(pts)
    while cur_len < pn:
        res = np.concatenate((res, pts))
        cur_len += pts.shape[0]
    return res[:pn, :]

def convert_label_to_one_hot(labels):
    label_one_hot = np.zeros((labels.shape[0], NUM_OBJ_CATS))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1
    return label_one_hot
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

def predict():
    is_training = False
    
    with tf.device('/gpu:'+str(gpu_to_use)):
        pointclouds_ph, input_label_ph = placeholder_inputs()
        is_training_ph = tf.placeholder(tf.bool, shape=())
        latent_ph = tf.placeholder(tf.float32, shape=(batch_size, 1,1,128))

        # simple model
        pred, seg_pred, end_points = model.get_model(pointclouds_ph, latent_ph,input_label_ph, \
                cat_num=NUM_OBJ_CATS, part_num=NUM_PART_CATS, is_training=is_training_ph, \
                batch_size=batch_size, num_point=point_num, weight_decay=0.0, bn_decay=None)
        
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        flog = open(os.path.join(output_dir, 'log.txt'), 'w')

        # Restore variables from disk.
        printout(flog, 'Loading model %s' % pretrained_model_path)
        saver.restore(sess, pretrained_model_path)
        printout(flog, 'Model restored.')
        
        # Note: the evaluation for the model with BN has to have some statistics
        # Using some test datas as the statistics
        batch_data = np.zeros([batch_size, point_num, 3]).astype(np.float32)

        total_acc = 0.0
        total_seen = 0
        total_acc_iou = 0.0

        total_per_cat_acc = np.zeros((NUM_OBJ_CATS)).astype(np.float32)
        total_per_cat_iou = np.zeros((NUM_OBJ_CATS)).astype(np.float32)
        total_per_cat_seen = np.zeros((NUM_OBJ_CATS)).astype(np.int32)

        ffiles = open(test_file_list, 'r')
        lines = [line.rstrip() for line in ffiles.readlines()]
        pts_files = [line.split()[0] for line in lines]
        seg_files = [line.split()[1] for line in lines]
        labels = [line.split()[2] for line in lines]
        ffiles.close()

        len_pts_files = latent.shape[0]
        shape_idx=0
        batch_sizes=batch_size
        while shape_idx <= len_pts_files:
            if shape_idx + batch_sizes <= len_pts_files:
                end_idx = shape_idx + batch_sizes
            else:
                break

            label_pred_val, seg_pred_res = sess.run([pred, seg_pred], feed_dict={
                        pointclouds_ph: np.zeros(((batch_size,2048,3))),
                        input_label_ph: np.zeros(((batch_size,2048,3))),
                        is_training_ph: is_training,
                        latent_ph:latentu
                        # latent_ph: latentu[shape_idx:end_idx]
                    })

            # label_pred_val = np.argmax(label_pred_val[0, :])


            if shape_idx == 0:
                pc_data = label_pred_val
            else:
                pc_data = np.vstack((pc_data, label_pred_val))
            shape_idx += batch_sizes
        np.save("pc_gen.npy", pc_data)

                
with tf.Graph().as_default():
    predict()
    b = np.load("pc_gen.npy")

    output_file = 'ex1_2.ply'
    one = np.ones((2048, 3))
    one = np.float32(one) * 255

    # Generate point cloud
    print("\n Creating the output file... \n")
    #    create_output(points_3D, colors, output_file)
    create_output(b[0], one, output_file)
