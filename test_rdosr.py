from tensorflow.python.client import device_lib
import tensorflow as tf
from data import *
from model_rdosr import*
import time
from os.path import join, exists
from os import makedirs, environ
import argparse
import scipy.io as sio
from datetime import datetime


parser = argparse.ArgumentParser(description='RDOSR_test')
parser.add_argument('--dataset', required=True, help='name of the used dataset')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()


#environ["CUDA_VISIBLE_DEVICES"] = "2"

# HSI datasets
dataset_dict = {'PaviaU':['paviaU',9], 'Pavia':['pavia',9], 'Indian_pines':['indian_pines',16]}

# set parameters
num_categories = dataset_dict[args.dataset][1]
lambda_r = 0.5
lambda_s = 0.001
lambda_c = 0.5

# type of layer design
decoder_layer = 10
hidden_layers = [3,3,3,3,3]


# load data
##known data
data = load_data('dataset_prep/'+args.dataset+'_kwn.mat')
x_kwn, y_kwn = format_data(data, key_x='x', key_y='y', num_labels=num_categories)
print('-- Known set x, y: ', x_kwn.shape, y_kwn.shape)

##unknown data
data = load_data('dataset_prep/'+args.dataset+'_unk.mat')
x_unk, y_unk = format_data(data, key_x='x', key_y='y', num_labels=num_categories)
print('-- Unknown set x, y: ', x_unk.shape, y_unk.shape)
x = np.concatenate((x_kwn,x_unk), axis=0)
y = np.concatenate((y_kwn,y_unk), axis=0)
n_samples, n_bands = x.shape


# create a folder for saving model and intermediate results
save_dir = 'results/'+ args.dataset + \
           '_' + str(decoder_layer) + \
	   '_s%.3f' % lambda_s
if not exists(save_dir):
    makedirs(save_dir)


# avoid allocating all memory of GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# build graph
tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_bands], name='input')
tf_y = tf.placeholder(dtype=tf.float32, shape=[None, num_categories], name='input_y')
network = Network(
    tf_x, tf_y, num_categories,
    args.lr, hidden_layers,
    decoder_layer, True, lambda_r, lambda_s, lambda_c, x.shape[0], False
)


# add ops to save and restore all the variable
saver = tf.train.Saver()

# the training process
print('---------> Testing on '+ args.dataset)
with tf.Session(config=config) as sess:
    # initialize the network
    tf.global_variables_initializer().run()
	
    # load pre-trained model
    load_file = tf.train.latest_checkpoint(save_dir)
    if load_file==None:
       print('Fail: Model was not found.') 
       exit()
    else:
       saver.restore(sess, load_file)
       print('SUCCESS: Model is loaded from %s\n' % load_file)


    # testing
    reconstruction_error = []
    y_ = []
    for i in range(n_samples):

        x_i = np.expand_dims(x[i], axis=0)
        y_i = np.expand_dims(y[i], axis=0)

        loss_EDC, loss_euc = sess.run(
                fetches=[network.loss_EDC, network.loss_euc],
                feed_dict={network.tf_x: x_i, network.tf_y: y_i}
                )
        print('Sample [%06d/%06d]: loss_euc = %.4f  loss_EDC = %.4f' % (i, n_samples, loss_euc, loss_EDC))

        reconstruction_error.append(loss_euc)

        if i<y_kwn.shape[0]:
            y_.append(np.argmax(y_i, axis=1)[0])
        else:
            y_.append(-1)      

    reconstruction_error = np.array(reconstruction_error)
    y_ = np.array(y_)

    sio.savemat(join(save_dir, 'outputs.mat'), {'recons_error':reconstruction_error, 'y':y_})





