import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  ### GPU is not used
import inputlayerclass as inputlayer
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
np.random.seed(0)


############### PREPARE THE SPIKE DATA ################## dataset can be mnist, emnist, cifar100,cifar10, caltech101, caltech256
T1 = time.time()
tsteps = 12
test_frac = 1.0/7
val_frac=test_frac
data_set = 'emnist'
labels_map = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',
              16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',
             31:'V',32:'W',33:'X',34:'Y',35:'Z',36:'a',37:'b',38:'d',39:'e',40:'f',41:'g',42:'h',43:'n',44:'q',45:'r',
             46:'t'}
l1_maps = 30
saved_set =   True
sample_interval=200
nofImages = 6000
intervals = nofImages/sample_interval
if(not saved_set):
            #### grabbing testing data ###########
    firstLayer = inputlayer.InputLayer(debug= not True,size=27,dataset=data_set,off_threshold=50,\
                                       on_threshold=50,border_size=2,data='test',val_frac = val_frac,test_frac =test_frac)
    test_input_data = firstLayer.EncodedData()   ### data is in this format [(data_tensor1, data_label1), (data_tensor2, data_label2), .......]


            #### grabbing training data ###########
    firstLayer = inputlayer.InputLayer(debug= not True,size=27,dataset=data_set,off_threshold=50,\
                                       on_threshold=50,border_size=2,data='train',val_frac = val_frac,test_frac = test_frac)
    train_input_data = firstLayer.EncodedData()   ### data is in this format [(data_tensor1, data_label1), (data_tensor2, data_label2), .......]


            #### grabbing validation data ###########
    firstLayer = inputlayer.InputLayer(debug= not True,size=27,dataset=data_set,off_threshold=50,\
                                       on_threshold=50,border_size=2,data='valid',val_frac = val_frac,test_frac = test_frac)
    valid_input_data = firstLayer.EncodedData()   ### data is in this format [(data_tensor1, data_label1), (data_tensor2, data_label2), .......]



            ##### combine the training and validation data so that the classifier class can split it inside the keras library #######
    train_input_data.extend(valid_input_data)
    random.shuffle(train_input_data)

           ###### convert the vectorized labels to integer values #######
    class_labels_train = map(lambda x: np.where(x[1]==1)[1][0], train_input_data)
    class_labels_test = map(lambda x: np.where(x[1]==1)[1][0], test_input_data)


    temp_input_data = [items[0] for items in train_input_data][0:nofImages]
    temp_input_data = np.concatenate(temp_input_data,axis=3)
    train_input_images = temp_input_data

def fillspikes(image):
    locs = np.where(image==1)
    image[locs[0],locs[1],:]=1
    return image


if(saved_set):
    
    filename ='emnist_train_on_x.h5'
    with h5py.File(filename, 'r') as hf:
        on_sparse = hf['on_time_tensor'][:]
    filename ='emnist_train_off_x.h5'
    with h5py.File(filename, 'r') as hf:
        off_sparse = hf['off_time_tensor'][:]
    on_sparse = np.split(on_sparse, range(12,on_sparse.shape[-1],tsteps), axis = 2)
    #on_sparse = map(fillspikes, on_sparse)
    on_sparse = np.concatenate(on_sparse, axis=-1)

    off_sparse = np.split(off_sparse, range(12,off_sparse.shape[-1],tsteps), axis = 2)
    #off_sparse = map(fillspikes, off_sparse)
    off_sparse = np.concatenate(off_sparse, axis=-1)
    train_input_images = np.zeros((27,27,2,12*112799),dtype = np.bool_)
    train_input_images[:,:,0,:] = on_sparse
    train_input_images[:,:,1,:] = off_sparse
    filehandle = open('emnist_train_y.pkl','rb')
    class_labels_train = pickle.load(filehandle).astype(np.int).tolist()
    filehandle.close()

plots = False

############## INITIALIZE AND TRAIN THE 1ST CONVOLUTION LAYER ############


nTrain_images = nofImages
net1 = network.Network(homeostasis_const=5, factor=3.0, pool_lateral_inh=False, output_channels=l1_maps, \
            inputs=train_input_images[:,:,:,0:12*nTrain_images], A_plus=0.002, debug=False, \
            sample_interval=sample_interval, train=True, save_pool_spike_tensor=False, save_pool_features=False,\
            homeo_supp_coeff=0.003, threshold=15.0,lr_inc_rate=500)
net1.feedforward()
if(plots):
    net1.feature_visualization([net1.evol_weights], sample_interval, intervals, plotx=5, ploty=6)
    net1.animation([net1.evol_weights], plotx=5, ploty=6, sample_interval=sample_interval, intervals=intervals)
    net1.feature_convergence([net1.evol_weights], sample_interval)
    net1.spike_statistics()

########## RE-INITIALIZE THE NETWORK WITH PREVIOUS WEIGHTS AND COLLECT TRIANING SPIKES IN POOL LAYER WITHOUT LATERAL INHIBITION#########
evolved_weights = net1.evol_weights[-1]
pool1_lateral =  False
net2 = network.Network(homeostasis_const=5, factor=5.0, pool_lateral_inh= pool1_lateral, inputs=train_input_images,\
                       output_channels=l1_maps,A_plus=0.002, train=False, set_weights=evolved_weights, debug=False,\
                       save_pool_spike_tensor=True,save_pool_features=True, threshold=15.0, pool_spike_accum=False)
net2.rewire_weights() ## fixing the weights
net2.feedforward()
if(plots):
    net2.spikes_per_map_per_class(plot_x=2,plot_y=2,class_labels=class_labels_train,pool_output_data=net2.pool_spike_tensor,\
                                  labels_map=labels_map, labelsize=14, view_maps=[24,25,26,27], final_weights=net1.evol_weights[-1])

train_pool_spike_features = net2.make_feature_vecs(net2.pool_spike_features)
train_pool_pot_features = net2.make_feature_vecs(net2.pool_pot_features)
sys.exit()
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


######## RE COLLECT THE SPIKES IN POOL1 BY TURNING ON THE LATERAL INHIBITION IN POOL1 ###############

######## RE-INITIALIZE THE NETWORK WITH PREVIOUS WEIGHTS AND COLLECT TRIANING SPIKES IN POOL LAYER #########
evolved_weights = net1.evol_weights[-1]
pool1_lateral =  True

evolved_weights = net1.evol_weights[-1]
net3 = network.Network(homeostasis_const=5, factor=5.0, pool_lateral_inh=pool1_lateral, inputs=train_input_images, output_channels=l1_maps,\
                       A_plus=0.002, train=False, set_weights=evolved_weights, debug=False, save_pool_spike_tensor=True,\
                       save_pool_features=False, threshold=15.0)
net3.rewire_weights() ## fixing the weights
net3.feedforward()
if(plots):
    net3.spikes_per_map_per_class(plot_x=2,plot_y=2,class_labels=class_labels_train,pool_output_data=net3.pool_spike_tensor,\
                                  labels_map=labels_map, labelsize=14, view_maps=[24,25,26,27], final_weights=net1.evol_weights[-1])


print('Total time taken:{}'.format(time.time()-T1))  


######## USE THE TRAIN POOLING SPIKES FROM THE PREVIOUS LAYER AND TRAIN THE 2ND CONVOLUTION LAYER (without pool inhibition in L3) ############ 
l4_maps = 200
pool2_lateral = False
size = net2.pool_spike_tensor.shape[0]
input_channels = net2.pool_spike_tensor.shape[2]
lr_inc_rate = 1500
nTrain_images = 35000 ##35k is enough if lateral inh in pool1 is False
net4 = network.Network(homeostasis_const=3,factor=3.0,pool_lateral_inh= pool2_lateral,inputs=net2.pool_spike_tensor[:,:,:,0:nTrain_images*tsteps],\
                       A_plus=0.0002,debug=False, output_channels=l4_maps, size=size, input_channels=input_channels,\
                       lr_inc_rate=lr_inc_rate,sample_interval=sample_interval,train=True,threshold=15.0,inh_reg=3,homeo_supp_coeff=0.0003,\
                       epochs=1)

net4.feedforward()

plots =  False
if(plots):
    net4.feature_convergence([net1.evol_weights, net4.evol_weights],sample_interval)
    net4.spike_statistics()
### feature reconstruction 
    layer_num = [2,3,4]
    filter_sizes = [net2.conv_kernel_size,net2.pool_kernel_size,net4.conv_kernel_size] ##[list of synapses from  first to last layers]
    filter_strides = [1,2,1]
    nof_filters = [net2.output_channels,net2.output_channels,net4.output_channels] ##[list of number of filters from first to last layers]
    types = ['conv','pool','conv']
    currLayer= 4
    layer_weights=[[net2.evol_weights],[net4.evol_weights]]
    net4.feature_visualization(layer_weights,sample_interval,intervals,plotx=5,ploty=5,layer_num=layer_num,\
                              filter_sizes=filter_sizes,filter_strides=filter_strides,nof_filters=nof_filters,\
                              types=types,currLayer=currLayer,show=True)
    net4.animation(layer_weights,sample_interval,intervals,plotx=10,ploty=10,layer_num=layer_num,\
                              filter_sizes=filter_sizes,filter_strides=filter_strides,nof_filters=nof_filters,\
                              types=types,currLayer=currLayer)

######## USE THE TRAIN POOLING SPIKES FROM THE PREVIOUS LAYER AND TRAIN THE 2ND CONVOLUTION LAYER (with pool inhibition in pool1) ######### 
l4_maps = 200
pool2_lateral = True
size = net3.pool_spike_tensor.shape[0]
input_channels = net3.pool_spike_tensor.shape[2]
lr_inc_rate = 1500
nTrain_images = 60000 ##35k is enough if lateral inh in pool1 is False
net5 = network.Network(homeostasis_const=3,factor=3.0,pool_lateral_inh= pool2_lateral,inputs=net3.pool_spike_tensor[:,:,:,0:nTrain_images*tsteps],\
                       A_plus=0.0002,debug=False, output_channels=l4_maps, size=size, input_channels=input_channels,\
                       lr_inc_rate=lr_inc_rate,sample_interval=sample_interval,train=True,threshold=15.0,inh_reg=3,\
                       homeo_supp_coeff=0.0003,epochs=2)

net5.feedforward()

plots =  False
if(plots):
    net5.feature_convergence([net1.evol_weights, net5.evol_weights],sample_interval)
    net5.spike_statistics()
### feature reconstruction 
    layer_num = [2,3,4]
    filter_sizes = [net3.conv_kernel_size,net3.pool_kernel_size,net5.conv_kernel_size] ##[list of synapses from  first to last layers]
    filter_strides = [1,2,1]
    nof_filters = [net3.output_channels,net3.output_channels,net5.output_channels] ##[list of number of filters from first to last layers]
    types = ['conv','pool','conv']
    currLayer= 4
    layer_weights=[[net3.evol_weights],[net5.evol_weights]]
    net5.feature_visualization(layer_weights,sample_interval,intervals,plotx=5,ploty=5,layer_num=layer_num,\
                              filter_sizes=filter_sizes,filter_strides=filter_strides,nof_filters=nof_filters,\
                              types=types,currLayer=currLayer,show=True)
    net5.animation(layer_weights,sample_interval,intervals,plotx=10,ploty=10,layer_num=layer_num,\
                              filter_sizes=filter_sizes,filter_strides=filter_strides,nof_filters=nof_filters,\
                              types=types,currLayer=currLayer)

######Fix the conv2 (L4) weights that were trained with spikes without lateral inhibition in L3 (pool1)
    ##and colelct spikes on pool2 (L5) without lateral inhibition #######

l4_maps = 200
pool2_lateral = False
evolved_weights = net4.evol_weights[-1]
size = net2.pool_spike_tensor.shape[0]
input_channels = net2.pool_spike_tensor.shape[2]
net6 = network.Network(pool_lateral_inh= pool2_lateral,inputs=net2.pool_spike_tensor,debug=False,output_channels=l4_maps,\
                       size=size, input_channels=input_channels, sample_interval=sample_interval, A_plus=0.0002, train=False,\
                       threshold=15.0,epochs=1, set_weights=evolved_weights, save_pool_spike_tensor=True)

net6.feedforward()

pltos = False
if(plots):
    net6.spikes_per_map_per_class(plot_x=2,plot_y=2,class_labels=class_labels_train,pool_output_data=net6.pool_spike_tensor,\
                                  labels_map=labels_map, labelsize=14, view_maps=[56, 64, 97, 167], final_weights=net4.final_features)


######Fix the conv2 (L4) weights that were trained with spikes with lateral inhibition in L3 (pool1)
    ##and colelct spikes on pool2 (L5) without lateral inhibition #######

l4_maps = 200
pool2_lateral = False
evolved_weights = net5.evol_weights[-1]
size = net3.pool_spike_tensor.shape[0]
input_channels = net3.pool_spike_tensor.shape[2]
net7 = network.Network(pool_lateral_inh= pool2_lateral,inputs=net3.pool_spike_tensor,debug=False,output_channels=l4_maps,\
                       size=size, input_channels=input_channels, sample_interval=sample_interval, A_plus=0.0002, train=False,\
                       threshold=15.0,epochs=1, set_weights=evolved_weights, save_pool_spike_tensor=True)

net7.feedforward()

pltos = False
if(plots):
    net7.spikes_per_map_per_class(plot_x=2,plot_y=2,class_labels=class_labels_train,pool_output_data=net6.pool_spike_tensor,\
                                  labels_map=labels_map, labelsize=14, view_maps=[35, 36, 51, 150], final_weights=net5.final_features)


