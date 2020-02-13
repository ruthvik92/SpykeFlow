import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'  ### GPU is not used
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='True'
import inputlayerclass as inputlayer
from spykeflow import network as network
from spykeflow import inputlayerclass as inputlayer
import Network as network
import time
import numpy as np
import random
import h5py
import sys
import pickle
from keras.utils import np_utils
import matplotlib
import matplotlib.animation as animation
#np.random.seed(0)


############### PREPARE THE SPIKE DATA ################## dataset can be mnist, emnist, cifar100,cifar10, caltech101, caltech256
T1 = time.time()
tsteps = 12
test_frac = 1.0/7
val_frac=test_frac
data_set = 'facemotor'

        #### grabbing testing data ###########
firstLayer = inputlayer.InputLayer(debug= not True, size=161, dataset=data_set, off_threshold=15,\
                                   on_threshold=15, border_size=2, data='test', val_frac=val_frac, test_frac=test_frac)
test_input_data = firstLayer.EncodedData()   ### data is in this format [(data_tensor1, data_label1), (data_tensor2, data_label2), .......]


        #### grabbing training data ###########
firstLayer = inputlayer.InputLayer(debug= not True, size=161, dataset=data_set, off_threshold=15,\
                                   on_threshold=15, border_size=2, data='train', val_frac=val_frac, test_frac=test_frac)
train_input_data = firstLayer.EncodedData()   ### data is in this format [(data_tensor1, data_label1), (data_tensor2, data_label2), .......]

        #### grabbing validation data ###########
firstLayer = inputlayer.InputLayer(debug= not True,size=161,dataset=data_set,off_threshold=15,\
                                   on_threshold=15,border_size=2,data='valid',val_frac = val_frac,test_frac = test_frac)
valid_input_data = firstLayer.EncodedData()   ### data is in this format [(data_tensor1, data_label1), (data_tensor2, data_label2), .......]



        ##### combine the training and validation data so that the classifier class can split it inside the keras library #######
train_input_data.extend(valid_input_data)
random.shuffle(train_input_data)
labels_map = {0:'Face', 1:'Motor'}

       ###### convert the vectorized labels to integer values #######

class_labels_train = map(lambda x: x[1].max(), train_input_data)
sample_interval=100
nofImages = len(train_input_data)
intervals = nofImages/sample_interval
temp_input_data = [items[0] for items in train_input_data][0:nofImages]
temp_input_data = np.concatenate(temp_input_data, axis=3)
train_input_images = temp_input_data
size = train_input_images.shape[0]
plots = False
############## INITIALIZE AND TRAIN THE 1ST CONVOLUTION LAYER ############
train_input_images[:,:,1,:] = 0 ##kill all off center spikes
l1_maps = 4
nTrain_images = 400
print('Number of training images:{}'.format(nTrain_images))
net1 = network.Network(homeostasis_const=1, factor=5.0, pool_lateral_inh= False, output_channels=l1_maps, \
            input_channels=2, inputs=train_input_images[:,:,:,0:12*nTrain_images], A_plus=0.002, debug=False, \
            sample_interval=sample_interval, train=True, save_pool_spike_tensor=False, save_pool_features=False,\
            homeo_supp_coeff=0.003, threshold=10.0, size=size, few_spikes=False, epochs=5, lr_inc_rate=500, weight_sigma=0.01)
net1.weights[:,:,:,1,:] = 0 ##kill all synapses for OFF center
net1.feedforward()
print('Time for encoding and training Conv1:{}'.format(time.time()-T1))  
if(True):
    net1.feature_visualization([net1.evol_weights],sample_interval,intervals=len(net1.evol_weights),plotx=2,ploty=2)
    net1.animation([net1.evol_weights],plotx=2,ploty=2,sample_interval=sample_interval,intervals=len(net1.evol_weights))
    net1.feature_convergence([net1.evol_weights],sample_interval)
    net1.spike_statistics()

T2 = time.time()
###################### RE-INITIALIZE THE NETWORK WITH PREVIOUS WEIGHTS AND COLLECT TRIANING SPIKES IN POOL LAYER #########
evolved_weights = net1.evol_weights[-1]
net2 = network.Network(pool_lateral_inh=True, inputs=train_input_images[:,:,:,0:12*nTrain_images], input_channels=2,\
                       output_channels=l1_maps,A_plus=0.002,train=False, set_weights=evolved_weights, \
                       debug=False, save_pool_spike_tensor=True, save_pool_features=True, threshold=10.0,\
                       pool_spike_accum=False, pool_kernel_size=7, size=size)
net2.rewire_weights() ## fixing the weights
net2.feedforward()
pool1_spikes = net2.pool_spike_tensor
print('Time for passing spikes through fixed Conv1-Pool1:{}'.format(time.time()-T2))
SAVE_DATA = False
if(SAVE_DATA):
    #h5file = 'emnist_train_pool1_acc_spikes_inh_'+str(pool1_lateral)+'_conv1maps_'+str(l1_maps)+'.h5'
    #with h5py.File(h5file, 'w') as hf: 
    #    hf.create_dataset("pool1_time_tensor",  data = net2.pool_spike_tensor)
    #print "pool1 train spike tensor .h5 file written to",h5file

    h5file = 'rot_emnist_train_pool1_acc_cont_spike_features_inh_'+str(pool1_lateral)+'_conv1maps_'+str(l1_maps)+'.h5'
    with h5py.File(h5file, 'w') as hf: 
        hf.create_dataset("pool1_spike_features",  data = train_pool_spike_features)
    print "pool1 train spike features .h5 file written to",h5file

    h5file = 'rot_emnist_train_pool1_acc_cont_pot_features_inh_'+str(pool1_lateral)+'_conv1maps_'+str(l1_maps)+'.h5'
    with h5py.File(h5file, 'w') as hf: 
        hf.create_dataset("pool1_pot_features",  data = train_pool_pot_features)
    print "pool1 train pot features .h5 file written to",h5file

#del net2
#del train_input_images
########################### USE THE TRAIN POOLING SPIKES FROM THE PREVIOUS LAYER AND TRAIN THE 2ND CONVOLUTION LAYER ############ (will not use for some datasets)
T3 = time.time()
sample_interval = 100
l4_maps = 20
size = pool1_spikes.shape[0]
input_channels = pool1_spikes.shape[2]
nTrain_images = nofImages 
net3 = network.Network(inputs=pool1_spikes,A_plus=0.0004, debug=False, output_channels=l4_maps, size=size,\
                       input_channels=input_channels, lr_inc_rate=500, sample_interval=sample_interval,\
                       train=True, threshold=60.0, inh_reg=5, epochs=20, few_spikes=False, conv_kernel_size=17)

net3.feedforward()
print('Time for training Conv2:{}'.format(time.time()-T3))



plots =  False
if(plots):
    net3.feature_convergence([net1.evol_weights,net3.evol_weights],sample_interval)
    net3.spike_statistics()
### feature reconstruction 
    layer_num = [2,3,4]
    filter_sizes = [net1.conv_kernel_size,net2.pool_kernel_size,net3.conv_kernel_size] ##[list of synapses from  first to last layers]
    filter_strides = [1,7,1]
    nof_filters = [net1.output_channels,net2.output_channels,net3.output_channels] ##[list of number of filters from first to last layers]
    types = ['conv','pool','conv']
    currLayer= 4
    layer_weights=[[net1.evol_weights],[net3.evol_weights]]
    plotx=10
    ploty=10
    net3.feature_visualization(layer_weights,sample_interval,intervals,plotx=4,ploty=5,layer_num=layer_num,\
                              filter_sizes=filter_sizes,filter_strides=filter_strides,nof_filters=nof_filters,\
                              types=types,currLayer=currLayer,show=True)
    net3.animation(layer_weights,sample_interval,intervals,plotx=4,ploty=5,layer_num=layer_num,\
                              filter_sizes=filter_sizes,filter_strides=filter_strides,nof_filters=nof_filters,\
                              types=types,currLayer=currLayer)


#####fix the conv2 weights and collect the spikes in pool2

l5_maps = 20
pool2_lateral = True
evolved_weights = net3.evol_weights[-1]
size = net2.pool_spike_tensor.shape[0]
input_channels = net2.pool_spike_tensor.shape[2]
net4 = network.Network(pool_lateral_inh= pool2_lateral,inputs=net2.pool_spike_tensor,debug=False,output_channels=l5_maps,\
                       size=size, input_channels=input_channels, train=False, threshold=60.0,epochs=1, conv_kernel_size=17,\
                       set_weights=evolved_weights, save_pool_spike_tensor=True)

net4.feedforward()

pltos = False
if(plots):
    net4.spikes_per_map_per_class(plot_x=2,plot_y=3,class_labels=class_labels_train,pool_output_data=net4.pool_spike_tensor,\
                                  labels_map=labels_map, labelsize=14, view_maps=[2, 3, 9, 13, 5, 6], final_weights=net3.final_features)



