import matplotlib.pyplot as plt
import numpy as np
import pyNN.nest as p
import relu_utils as alg
import spiking_relu as sr
import random
import mnist_utils as mu
import os.path
import sys
import cnn_utils as cnnu
import matplotlib.cm as cm
import spiking_cnn as scnn

import scipy.io as sio
tmp_x = sio.loadmat('mnist.mat')['test_x']
tmp_x = np.transpose(tmp_x, (2, 0, 1))
tmp_x = np.reshape(tmp_x, (tmp_x.shape[0], 28*28), order='F' )

tmp_y = sio.loadmat('mnist.mat')['test_y']
tmp_y = np.argmax(tmp_y, axis=0)
dur_test = 1000
silence = 200
num_test = 100 
max_rate = 0
'''
cell_params_lif = {'cm': 0.25,
                   'tau_m': 20.0,
                   'tau_refrac': 1.,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }
'''
cell_params_lif = {'cm': 0.25,      #nF
                   'i_offset': 0.1, #nA
                   'tau_m': 20.0,   #ms
                   'tau_refrac': 1.,#ms
                   'tau_syn_E': 5.0,#ms
                   'tau_syn_I': 5.0,#ms
                   'v_reset': -65.0,#mV
                   'v_rest': -65.0, #mV
                   'v_thresh': -50.0#mV
                   }                   
#w_cnn, l_cnn = cnnu.readmat('scale_noisy_softplus_30.mat')#cnn609.mat softplus 3-5 train. 'cnn_relu.mat' 'scale_softplus.mat'
w_cnn, l_cnn = cnnu.readmat('scaled.mat')# scaled is the 0.023 nsp training.
predict = np.zeros(10000)
for offset in range(0, 10000, num_test):
    test = tmp_x[offset:(offset+num_test), :]
    test = test * 30.
    predict[offset:(offset+num_test)],  spikes= scnn.scnn_test(cell_params_lif, l_cnn, w_cnn, num_test, test, max_rate, dur_test, silence)
    print sum(predict[offset:(offset+num_test)]==tmp_y[offset:(offset+num_test)]) 
    spike_f = 'spikes_nsp/spike_scaled_%d.npy'%(offset)
    np.save(spike_f, spikes)
np.save('predict_result',predict) 
print sum(predict==tmp_y)

