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


parser = argparse.ArgumentParser(description='RDOSR_train')
parser.add_argument('--dataset', required=True, help='name of the used dataset')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=15000, help='number of epochs')
parser.add_argument('--ratio', type=float, default=0.1, help='ratio between training data and testing data')
args = parser.parse_args()


#environ["CUDA_VISIBLE_DEVICES"] = "2"

# HSI datasets
dataset_dict = {'PaviaU':['paviaU',9], 'Pavia':['pavia',9], 'Indian_pines':['indian_pines',16]}

# set parameters
num_categories = dataset_dict[args.dataset][1]
num_epochs = args.n_epochs
tol = 0.99
lambda_r = 0.5
lambda_s = 0.001
lambda_c = 0.5

# type of layer design
decoder_layer = 10
hidden_layers = [3,3,3,3,3]


# load data
data = load_data('dataset_prep/'+args.dataset+'_kwn.mat')
x, y = format_data(data, key_x='x', key_y='y', num_labels=num_categories)

# split data into training and testing sets
x, y, x_te, y_te = seed_train_test(x, y, percentage=args.ratio)
print('--x, y, x_te, y_te: ', x.shape, y.shape, x_te.shape, y_te.shape)
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

# build graph
tf_x_te = tf.placeholder(dtype=tf.float32, shape=[None, n_bands], name='input')
tf_y_te = tf.placeholder(dtype=tf.float32, shape=[None, num_categories], name='input_y')
network_te = Network(
    tf_x_te, tf_y_te, num_categories,
    args.lr, hidden_layers,
    decoder_layer, True, lambda_r, lambda_s, lambda_c, x_te.shape[0], True
)


# add ops to save and restore all the variable
saver = tf.train.Saver()

# the training process
print('---------> Training on '+ args.dataset)
with tf.Session(config=config) as sess:
    # initialize the network
    tf.global_variables_initializer().run()
	
    # load pre-trained model
    if os.path.exists(save_dir):
        load_file = tf.train.latest_checkpoint(save_dir)
        if load_file==None:
            print('FAIL: No checkpoint was saved.\n')
        else:
            saver.restore(sess, load_file)
            print('SUCCESS: Model is loaded from %s\n' % load_file)

    # log testing loss and testing accuracy
    with open(join(save_dir, 'history.log'), 'a') as f:
        f.write('[%s] %s\n' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.dataset))

    loss_F, acc_F, loss_EDC, loss_euc = 0, 0, 0, 0
    # iterate epochs
    for epoch in range(num_epochs):
		
        if acc_F < tol:
            # update the network F
            _, loss_F, acc_F = sess.run(
                fetches=[network.opt_F, network.loss_F, network.accuracy_F],
                feed_dict={network.tf_x: x, network.tf_y: y}
            )
		
            print('Epoch [%03d/%03d]\n'
                  '\ttraining: loss_F = %.4f  acc_F = %.4f'%
                  (epoch + 1, num_epochs, loss_F, acc_F))
	
        else:	
            # update the network EDC
            _, loss_EDC, loss_euc = sess.run(
                fetches=[network.opt_EDC, network.loss_EDC, network.loss_euc],
                feed_dict={network.tf_x: x, network.tf_y: y}
            )

            print('Epoch [%03d/%03d]\n'
                  '\ttraining: loss_EDC = %.4f  loss_euc = %.4f'%
                  (epoch + 1, num_epochs, loss_EDC, loss_euc))

        # get testing results
        loss_F_te, acc_F_te, loss_EDC_te, loss_euc_te = sess.run(
                fetches=[network_te.loss_F, network_te.accuracy_F, network_te.loss_EDC, network_te.loss_euc],
                feed_dict={network_te.tf_x: x_te, network_te.tf_y: y_te}
            )
        #print('--> Epoch [%03d/%03d]\n'
        print('     \ttesting:  loss_F = %.4f  acc_F = %.4f  loss_EDC = %.4f  loss_euc = %.4f'%
                  (loss_F_te, acc_F_te, loss_EDC_te, loss_euc_te))


        # write testing results to log file
        with open(join(save_dir, 'history.log'), 'a') as f:
            f.write('%04d,%.4f,%.4f,%.4f,%.4f\n' % (epoch + 1, loss_F_te, acc_F_te, loss_EDC_te, loss_euc_te))

        # save the model
        if (epoch+1) % 100 == 0:
            save_path = saver.save(sess, join(save_dir, 'model.ckpt'))
            print('\nSUCCESS: Model is saved to %s\n' % save_path)


