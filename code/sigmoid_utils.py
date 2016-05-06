'''
Functions to be used in Sigmoid tasks.
'''

import mnist_utils as mu
import maths_utils as matu
import numpy as np
from scipy.special import expit
import pickle

def save_dict(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
        
def init_para(vis_num, hid_num, eta):
    para = {}
    para['h_num'] = hid_num
    para['v_num'] = vis_num
    para['eta'] = eta
    w = np.random.normal(0,0.01, vis_num*hid_num)
    para['w'] = w.reshape((vis_num,hid_num))
    para['b'] = np.zeros(hid_num)
    para['a'] = np.zeros(vis_num)
    #pixel_on = np.sum(Data_v,0)
    #para['a'] = np.log((pixel_on + 0.01)/(train_num - pixel_on + 0.01))
    return para

def plot_recon(digit_img, para):
    data_v = np.array(digit_img).astype(float)
    data_h, gibbs_v, gibbs_h = sampling_nb(para, data_v)
    mu.plot_digit(gibbs_v)
    
def update_batch_cd1(para, data_v, layer=1):
    eta = para['eta']
    max_bsize = data_v.shape[0]
    if layer == 0: # input layer, otherwise they are binary
        data_h, gibbs_v, gibbs_h = sampling_nb(para, data_v)
    else:
        data_h, gibbs_v, gibbs_h = sampling(para, data_v)
    
    pos_delta_w = np.zeros((para['v_num'], para['h_num']))
    neg_delta_w = np.zeros((para['v_num'], para['h_num']))
    for i in range(max_bsize):
        pos_delta_w += matu.matrix_times(data_v[i], data_h[i])
        neg_delta_w += matu.matrix_times(gibbs_v[i], gibbs_h[i])    
    delta_w_pos = eta * pos_delta_w/np.float(max_bsize)
    delta_w_neg = eta * neg_delta_w/np.float(max_bsize)
    para['w'] += delta_w_pos
    para['w'] -= delta_w_neg
    delta_a = data_v - gibbs_v
    delta_b = data_h - gibbs_h
    delta_a = eta * np.average(delta_a,0)
    delta_b = eta * np.average(delta_b,0)
    para['a'] += delta_a
    para['b'] += delta_b
    #print delta_w_pos.max(), delta_w_neg.max()
    return para
    

    
def sigmoid(data, weight, bias):
    sum_data = np.dot(data, weight) + bias
    prob = expit(sum_data)
    return prob



def sigmoid_sampling(data, weight, bias):
    prob = sigmoid(data, weight, bias)
    rdm = np.random.random(prob.shape)
    index_on = rdm < prob
    samples = np.zeros(prob.shape)
    samples[index_on]=1.
    return samples
  
def sampling(para, data_v): #binary
    w = para['w']
    a = para['a']
    b = para['b']
    h0 = sigmoid_sampling(data_v, w, b)
    v1 = sigmoid_sampling(h0, w.transpose(), a)
    h1 = sigmoid_sampling(v1, w, b)

    return h0, v1, h1  

def sampling_nb(para, data_v): #non binary
    w = para['w']
    a = para['a']
    b = para['b']
    h0 = sigmoid_sampling(data_v, w, b)
    v1 = sigmoid(h0, w.transpose(), a)
    h1 = sigmoid_sampling(v1, w, b)

    return h0, v1, h1
    
def init_label_dbn(train_data, label_data, nodes, eta=1e-3, batch_size=10, epoc=5):
    if train_data.shape[1] != nodes[0]:
        print 'Dimention of train_data has to equal to the input layer size.'
        exit()
    elif label_data.shape[1] != nodes[-1]:
        print 'Dimention of label_data has to equal to the output layer size.'
        exit()
    elif train_data.shape[0] != label_data.shape[0]:
        print 'The amount of data and label should be the same.'
        exit()
    print epoc
    dbnet = {}
    dbnet['train_x'] = train_data
    dbnet['train_y'] = label_data
    dbnet['nodes'] = nodes
    dbnet['batch_size'] = batch_size
    dbnet['epoc'] = epoc
    
    para_list = []
    for i in range(len(nodes) - 3):   #bottom up
        para_list.append(init_para(nodes[i], nodes[i+1], eta))
    para_top = init_para(nodes[-3] + nodes[-1], nodes[-2], eta)
    para_top['label_n'] = 10
    dbnet['layer'] = para_list
    dbnet['top'] = para_top
    
    return dbnet
    
def RBM_train(para, layer_num, epoc, batch_size, train_data):
    train_num = train_data.shape[0]
    for iteration in range(epoc):
        for k in range(0,train_num,batch_size):
            max_bsize = min(train_num-k, batch_size)
            data_v = train_data[k:k+max_bsize]
            para = update_batch_cd1(para, data_v, layer=layer_num)
    return para
    
def greedy_train(dbnet):
    batch_size = dbnet['batch_size']
    train_data = dbnet['train_x']
    for i in range(len(dbnet['layer'])):   #bottom up
        dbnet['layer'][i] = RBM_train(dbnet['layer'][i], i, dbnet['epoc'], batch_size, train_data)
        train_data = sigmoid_sampling(train_data, dbnet['layer'][i]['w'], dbnet['layer'][i]['b'])
    train_data = np.append(train_data, dbnet['train_y'], axis=1)
    dbnet['top'] = RBM_train(dbnet['top'], i, dbnet['epoc'], batch_size, train_data)
    return dbnet


def update_unbound_w(w_up, w_down, b_in, b_out, d_vis, layer=1):
    bsize = d_vis.shape[0]
    delta_w = 0
    d_hid = sigmoid_sampling(d_vis, w_up, b_out)
    if layer == 0:
        g_vis = sigmoid(d_hid, w_down, b_in)
    else:
        g_vis = sigmoid_sampling(d_hid, w_down, b_in)
    
    for ib in range(bsize):
        delta_w += matu.matrix_times(d_hid[ib], d_vis[ib]-g_vis[ib])
    delta_w = delta_w/np.float(bsize)
    delta_b = np.average(d_vis[ib]-g_vis[ib], axis=0)
    return delta_w, delta_b, d_hid
    
def fine_train(dbnet):
    batch_size = dbnet['batch_size']
    train_data = dbnet['train_x']
    train_num = train_data.shape[0]
    for i in range(len(dbnet['layer'])):   #bottom up
        dbnet['layer'][i]['w_up'] = dbnet['layer'][i]['w']
        dbnet['layer'][i]['w_down'] = np.transpose(dbnet['layer'][i]['w'])
    for iteration in range(dbnet['epoc']):
        for k in range(0,train_num,batch_size):
            max_bsize = min(train_num-k, batch_size)
            d_vis = train_data[k:k+max_bsize]
            label = dbnet['train_y'][k:k+max_bsize]
            
            #up
            for i in range(len(dbnet['layer'])):   #bottom up
                delta_w, delta_b, d_vis = update_unbound_w(dbnet['layer'][i]['w_up'], dbnet['layer'][i]['w_down'], dbnet['layer'][i]['a'], dbnet['layer'][i]['b'], d_vis, layer=i)
                dbnet['layer'][i]['w_down'] += dbnet['layer'][i]['eta'] * delta_w
                dbnet['layer'][i]['a'] += dbnet['layer'][i]['eta'] * delta_b
            #top
            d_vis = np.append(d_vis, label, axis=1)
            dbnet['top'] = update_batch_cd1(dbnet['top'], d_vis) #layer defult == 1 #binary
            d_hid, g_vis, g_hid = sampling(dbnet['top'], d_vis)
            d_vis = g_vis[:, :dbnet['top']['v_num'] - dbnet['top']['label_n']]
            #down
            for i in range(len(dbnet['layer'])-1, -1, -1):   #up down
                delta_w, delta_b, d_vis = update_unbound_w(dbnet['layer'][i]['w_down'], dbnet['layer'][i]['w_up'], dbnet['layer'][i]['b'], dbnet['layer'][i]['a'], d_vis, layer=i)
                dbnet['layer'][i]['w_up'] += dbnet['layer'][i]['eta'] * delta_w
                dbnet['layer'][i]['b'] += dbnet['layer'][i]['eta'] * delta_b

    return dbnet
    
def dbn_recon(dbnet, test):
    temp = test
    top_inputsize = dbnet['top']['v_num'] - dbnet['top']['label_n']
    for i in range(len(dbnet['layer'])):   #bottom up
        temp = sigmoid_sampling(temp, dbnet['layer'][i]['w_up'], dbnet['layer'][i]['b'])
    top = sigmoid_sampling(temp, dbnet['top']['w'][:top_inputsize, :], dbnet['top']['b'])
    label = sigmoid_sampling(top, np.transpose(dbnet['top']['w'][top_inputsize:, :]), dbnet['top']['a'][top_inputsize:])
    temp = np.append(temp, label, axis=1)
    temp = sigmoid_sampling(temp, dbnet['top']['w'], dbnet['top']['b'])
    temp = sigmoid_sampling(temp, np.transpose(dbnet['top']['w']), dbnet['top']['a'])
    temp = temp[:top_inputsize]
    for i in range(len(dbnet['layer'])-1, 0, -1):   #up down
        temp = sigmoid_sampling(temp, dbnet['layer'][i]['w_down'], dbnet['layer'][i]['a'])
    recon = sigmoid(temp, dbnet['layer'][0]['w_down'], dbnet['layer'][0]['a'])
    mu.plot_digit(recon)
    predict = np.argmax(label)
    return predict, recon

def greedy_recon(dbnet, test): 
    temp = test
    top_inputsize = dbnet['top']['v_num'] - dbnet['top']['label_n']
    for i in range(len(dbnet['layer'])):   #bottom up
        temp = sigmoid_sampling(temp, dbnet['layer'][i]['w'], dbnet['layer'][i]['b'])
    top = sigmoid_sampling(temp, dbnet['top']['w'][:top_inputsize, :], dbnet['top']['b'])
    label = sigmoid_sampling(top, np.transpose(dbnet['top']['w'][top_inputsize:, :]), dbnet['top']['a'][top_inputsize:])
    temp = np.append(temp, label, axis=1)
    temp = sigmoid_sampling(temp, dbnet['top']['w'], dbnet['top']['b'])
    temp = sigmoid_sampling(temp, np.transpose(dbnet['top']['w']), dbnet['top']['a'])
    temp = temp[:top_inputsize]
    for i in range(len(dbnet['layer'])-1, 0, -1):   #up down
        temp = sigmoid_sampling(temp, np.transpose(dbnet['layer'][i]['w']), dbnet['layer'][i]['a'])
    recon = sigmoid(temp, np.transpose(dbnet['layer'][0]['w']), dbnet['layer'][0]['a'])
    mu.plot_digit(recon)
    predict = np.argmax(label)
    return predict, recon
    
def test_label_data(dbnet, test_data, test_label):
    dbnet['test_x'] = test_data
    dbnet['test_y'] = test_label
    return dbnet

def dbn_test(dbnet):
    temp = dbnet['test_x']
    top_inputsize = dbnet['top']['v_num'] - dbnet['top']['label_n']
    for i in range(len(dbnet['layer'])):   #bottom up
        temp = sigmoid_sampling(temp, dbnet['layer'][i]['w'], dbnet['layer'][i]['b'])
    top = sigmoid_sampling(temp, dbnet['top']['w'][:top_inputsize, :], dbnet['top']['b'])
    label = sigmoid_sampling(top, np.transpose(dbnet['top']['w'][top_inputsize:, :]), dbnet['top']['a'][top_inputsize:])
    
    predict = np.argmax(label, axis=1)
    index = np.where(label.max(axis=1)==0)[0]
    predict[index] = -1
    result = predict == dbnet['test_y']
    result = result.astype(int)
    result[index] = -1
    return predict, result

def dbn_greedy_test(dbnet):
    temp = dbnet['test_x']
    top_inputsize = dbnet['top']['v_num'] - dbnet['top']['label_n']
    for i in range(len(dbnet['layer'])):   #bottom up
        temp = sigmoid_sampling(temp, dbnet['layer'][i]['w'], dbnet['layer'][i]['b'])
    top = sigmoid_sampling(temp, dbnet['top']['w'][:top_inputsize, :], dbnet['top']['b'])
    label = sigmoid_sampling(top, np.transpose(dbnet['top']['w'][top_inputsize:, :]), dbnet['top']['a'][top_inputsize:])
    
    predict = np.argmax(label, axis=1)
    index = np.where(label.max(axis=1)==0)[0]
    predict[index] = -1
    result = predict == dbnet['test_y']
    result = result.astype(int)
    result[index] = -1
    return predict, result
