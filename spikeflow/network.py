import numpy as np
import poolconvmapclass as poolconv
import regionclass as supp_region
from itertools import izip as zip
import matplotlib.pyplot as plt
import matplotlib
import progressbar
import sys,os
import tensorflow as tf
import pickle
import cv2
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from copy import deepcopy
import matplotlib.animation as animation
import theano
font_size = 32
label_size = 24
dtype = np.float32
tfdtype = tf.float32
sess= tf.Session(config=tf.ConfigProto(log_device_placement=not True))
matplotlib.rcParams.update({'font.size': font_size})
#np.random.seed(0)
##os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

class Network(object):
    """ \n This class has methods for convolution and pooling and it can handle turning lateral inhibition ON or OFF and
        it also provides methods for feature visualization and spike statstics. debug feature let's you visualize the
        spike activity in convolution and pooling layers under the influence of stdp competition and lateral inhibiiton.
    """

    def __init__(self,inputs=None,A_plus=0.004,conv_kernel_size=5,pool_kernel_size=2,\
                 threshold=15.0,conv_lateral_inh=True,pool_lateral_inh= False,STDP_compet=True, size=27, input_channels = 2,\
                 output_channels = 30, tsteps = 12,homeostasis_const=5,factor=4.0,homeo_supp_coeff=0.003,inh_reg=11,debug= False, train=True,\
                 sample_interval = 500, lr_inc_rate = 1000, set_weights=None, save_pool_features=True,\
                 save_pool_spike_tensor=False, epochs=1,pool_spike_accum=True, few_spikes=True, weight_sigma=0.01):
        
        self.epochs = epochs
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.save_pool_features = save_pool_features
        self.save_pool_spike_tensor = save_pool_spike_tensor
        self.threshold = threshold
        self.conv_lateral_inh = conv_lateral_inh
        self.pool_lateral_inh = pool_lateral_inh
        self.STDP_compet = STDP_compet
        self.pool_spike_accum = pool_spike_accum
        self.size=size
        self.output_size = self.size-self.conv_kernel_size+1
        self.input_channels = input_channels
        self.output_channels = output_channels
        ##############Setting up convoltuion graph (tensorflow actually does correlations)
        self.image_placeholder = tf.placeholder(tfdtype, shape=[1,self.size,self.size,self.input_channels])
        self.kernel_placeholder = tf.placeholder(tfdtype, shape=[self.conv_kernel_size,self.conv_kernel_size,\
                                                                    self.input_channels,self.output_channels])
        self.conv_strides_1d = [1, 1, 1, 1]
        self.inter = tf.squeeze(tf.nn.conv2d(self.image_placeholder, self.kernel_placeholder, strides=self.conv_strides_1d, \
                                             padding='VALID'))
        
        ##############Setting up pooling graph
        self.pool_image_placeholder = tf.placeholder(tfdtype,shape=[1,self.output_size,self.output_size,self.output_channels])
        self.pool_strides = [1, self.pool_kernel_size, self.pool_kernel_size,1]
        self.pool_size    = [1, self.pool_kernel_size, self.pool_kernel_size,1]
        self.inter_pooling = tf.squeeze(tf.nn.max_pool(self.pool_image_placeholder, ksize =self.pool_size,strides=self.pool_strides,\
                                            padding='VALID',data_format= 'NHWC'))
        self.init_op = tf.global_variables_initializer()
        sess.run(self.init_op)
        ################################# end of pooling and convolution graph settings
        #self.weights = np.zeros((self.output_size,self.output_size,self.conv_kernel_size**2,self.input_channels,\
        #                         self.output_channels))
        self.weights = np.zeros((1,1,self.conv_kernel_size**2,self.input_channels,\
                                 self.output_channels))
        self.evol_weights = []
        self.weight_mu = 0.8
        self.weight_sigma = weight_sigma
        self.delay = 1.0
        self.tsteps = tsteps
        self.pots = np.zeros((self.output_size, self.output_size, self.output_channels),dtype)
        self.pots_ = np.zeros((self.output_size, self.output_size, self.output_channels),dtype)
        self.StdpPerNeuronAllowed = np.ones((self.output_size,self.output_size,self.output_channels),dtype)
        self.SpikesPerNeuronAllowed = np.ones((self.output_size,self.output_size),dtype)
        self.pool_SpikesPerNeuronAllowed = np.ones((self.output_size/self.pool_kernel_size,\
                                                    self.output_size/self.pool_kernel_size),dtype)
        self.conv_spikes = np.zeros((self.output_size,self.output_size,self.output_channels),dtype=dtype)
        self.pool_spikes = np.zeros((self.output_size/self.pool_kernel_size,self.output_size/self.pool_kernel_size,\
                                     self.output_channels),dtype)
        self.pool_pots = np.zeros((self.output_size/self.pool_kernel_size, self.output_size/self.pool_kernel_size, \
                                   self.output_channels),dtype)
        
        self.input = inputs
        self.sim_time = self.input.shape[-1]
        self.nofImages = self.sim_time/self.tsteps
        self.train = train
        self.homeo_supp_coeff = homeo_supp_coeff
        #if(self.save_pool_spike_tensor):
        if(not self.train and self.save_pool_spike_tensor):
            self.pool_spike_tensor = np.zeros((self.output_size/self.pool_kernel_size,self.output_size/self.pool_kernel_size,\
                                     self.output_channels,self.sim_time),np.bool_)
        #self.pool_pot_tensor = np.zeros((self.output_size/self.pool_kernel_size,self.output_size/self.pool_kernel_size,\
        #                             self.output_channels,self.sim_time),np.float32)
        #if(self.save_pool_features):
        if(not self.train and self.save_pool_features):
            self.pool_pot_features = np.zeros((self.output_size/self.pool_kernel_size,self.output_size/self.pool_kernel_size,\
                                     self.output_channels,self.nofImages),np.float32)
            self.pool_spike_features = np.zeros((self.output_size/self.pool_kernel_size,self.output_size/self.pool_kernel_size,\
                                     self.output_channels,self.nofImages),np.int16)
        #self.pool_pot_features = [np.zeros(self.pool_pots.flatten().shape)]*(self.sim_time/self.tsteps)
        self.final_stdp_ids = None
        
        self.spiked_neurons= None
        self.pool_spiked_neurons= None
        self.updates_per_map = [1]*self.output_channels
        self.updates_per_map_imgs = [0]*self.output_channels
        self.homeostasis_const = homeostasis_const
        self.factor = factor
        self.inh_reg = inh_reg
        self.offset = inh_reg/2
        self.final_weights = np.zeros((self.conv_kernel_size,self.conv_kernel_size,3,self.output_channels))
        self.stdp_spikes=[]
        self.debug = debug
        self.A_plus = A_plus
        self.final_features =None #note that 0th axis is OFF and 1st is ON for RGB viz. this is opposite to self.weights
        self.A_minus = self.A_plus*0.75
        self.sample_interval = sample_interval
        self.lr_inc_rate =  lr_inc_rate
        self.set_weights = set_weights
        self.few_spikes = few_spikes
        print('Setting up the synapses and the weights\n')
        self.connectivity=self.network_setup()
        print('connections of first neuron in the maps to neurons in input image(prev_map):\n{}\n'.format(self.connectivity[0]))        
        print('connections of last neuron in the maps to neurons in input image(prev_map):\n{}\n'.format(self.connectivity[-1]))
        #print('There are:{} neurons in each of the input map and there are:{} synapses from previous maps going into each\
        #neuron of the input map and there are:{} previous maps and there are:{} input maps'.format(self.weights.shape[0:2],\
        #self.weights.shape[2],self.weights.shape[3],self.weights.shape[4]))
        print('There are:{} neurons in each of the input map and there are:{} synapses from previous maps going into each\
        neuron of the input map and there are:{} previous maps and there are:{} input maps'.format((self.size,self.size),\
        self.weights.shape[2],self.weights.shape[3],self.weights.shape[4]))



    def network_setup(self):
        '''param: nofMaps: is the number of output maps
        param: kernel_size: is the size of the kernel
        param: image_size: is the size of the input map(image)
        This function sets up connections, weights between input map and output maps
        and returns an array of mappings between input map and output maps.
        '''
        #@for j in range(self.output_channels):
        #@    for k in range(self.input_channels):
        #@        a=poolconv.poolconv(self.conv_kernel_size,self.size,self.weight_mu,self.weight_sigma,self.delay,\
        #@                            self.conv_kernel_size-1)
        #@        indices, map_weights = a.IndicesWeights()
        #@        self.weights[:,:,:,k,j] = (np.asarray(map_weights)).reshape((self.output_size,self.output_size,\
        #@                                                                     self.conv_kernel_size**2))
        a=poolconv.poolconv(self.conv_kernel_size,self.size,self.weight_mu,self.weight_sigma,self.delay,\
                            self.conv_kernel_size-1)
        indices,_ = a.IndicesWeights()
        temp = np.random.normal(self.weight_mu, self.weight_sigma, self.conv_kernel_size**2*self.input_channels* \
        self.output_channels)
        temp = temp.reshape(1,1,self.conv_kernel_size**2, self.input_channels, self.output_channels)
        temp[np.where(temp>=1.0)] = 0.8
        temp[np.where(temp<=0.0)] = 0.8
        self.weights = temp.astype(dtype) 
        return indices

    def rewire_weights(self):
        temp = self.set_weights.reshape(self.conv_kernel_size**2,self.input_channels,-1)
        self.weights = temp[np.newaxis, np.newaxis, :]
        #self.weights = np.tile(temp,(self.output_size,self.output_size,1,1,1))
        self.evol_weights.append(self.set_weights)

        return

    def convolution(self, input_image):

        Weights = self.weights[0,0,:,:,:]
        Weights = Weights.reshape(self.conv_kernel_size,self.conv_kernel_size,self.input_channels,self.output_channels)
        input_4d = input_image.reshape(1,self.size,self.size,self.input_channels)
        kernel_4d = Weights
        output_3d= sess.run(self.inter,feed_dict ={self.image_placeholder:input_4d,self.kernel_placeholder:kernel_4d})

        return output_3d

    def pooling(self,conv_pots):
        #print(conv_pots.shape)
        conv_pots = conv_pots[np.newaxis,:,:,:]
        output_3d= sess.run(self.inter_pooling,feed_dict ={self.pool_image_placeholder:conv_pots})
        #print('after:{}'.format(output_3d.dtype))

        return output_3d

    def scipy_pooling(self,conv1_spikes):
        '''
           Reshape splitting each of the two axes into two each such that the
           latter of the split axes is of the same length as the block size.
           This would give us a 4D array. Then, perform maximum finding along those
           latter axes, which would be the second and fourth axes in that 4D array.
           (https://stackoverflow.com/questions/41813722/numpy-array-reshaped-but-how-to-change-axis-for-pooling
        '''
        if(conv1_spikes.shape[0]%2!=0): #if array is odd size then omit the last row and col
            conv1_spikes = conv1_spikes[0:-1,0:-1,:]
        else:
            conv1_spikes = conv1_spikes
        m,n = conv1_spikes[:,:,0].shape
        o   = conv1_spikes.shape[-1]
        pool1_spikes = np.zeros((m/2,n/2,o))
        for i in range(o):
            pool1_spikes[:,:,i]=conv1_spikes[:,:,i].reshape(m/2,2,n/2,2).max(axis=(1,3))
        return pool1_spikes

    def scipy_convolution(self,Image,weights):
        instant_conv1_pots = np.zeros((self.size-self.conv_kernel_size+1,self.size-self.conv_kernel_size+1,self.output_channels))
        for i in range(weights.shape[-1]):
            instant_conv1_pots[:,:,i]=cv2.filter2D(Image.astype(np.float64),-1,weights[:,:,i])[2:-2,2:-2]
        return instant_conv1_pots
    
    def reset(self,image_number):
        self.pots = np.zeros((self.output_size, self.output_size, self.output_channels),dtype = dtype)
        self.pots_ = np.zeros((self.output_size, self.output_size, self.output_channels),dtype = dtype)
        self.StdpPerNeuronAllowed = np.ones((self.output_size,self.output_size,self.output_channels),dtype = dtype)
        self.SpikesPerNeuronAllowed = np.ones((self.output_size,self.output_size),dtype = dtype)
        self.pool_pots = np.zeros((self.output_size/self.pool_kernel_size, self.output_size/self.pool_kernel_size, \
                                   self.output_channels),dtype = dtype)
        self.pool_spike_tensor = np.zeros((self.output_size/self.pool_kernel_size,self.output_size/self.pool_kernel_size,\
                                     self.output_channels,self.tsteps),dtype = dtype)
        #self.pool_spike_tensor = np.zeros((self.output_size,self.output_size,\
        #                             self.output_channels,self.tsteps),dtype = dtype)
        self.pool_SpikesPerNeuronAllowed = np.ones((self.output_size/self.pool_kernel_size,\
                                                    self.output_size/self.pool_kernel_size),dtype = dtype)
        self.conv_spikes = np.zeros((self.output_size,self.output_size,self.output_channels),dtype=dtype)

        self.pool_spikes = np.zeros((self.output_size/self.pool_kernel_size,self.output_size/self.pool_kernel_size,\
                                     self.output_channels),dtype)
        #self.pool_spikes = np.zeros((self.output_size,self.output_size,\
        #                             self.output_channels),dtype)
        if(image_number%self.homeostasis_const==0):
            self.updates_per_map_imgs = [0]*self.output_channels


    def feedforward(self):
        
        ip_img = np.zeros((self.size,self.size))
        spikes_pre_inh = np.zeros((self.output_size,self.output_size,self.output_channels))
        spikes_post_inh = np.zeros((self.output_size,self.output_size,self.output_channels))
        spikes_post_stdp = np.zeros((self.output_size,self.output_size,self.output_channels))
        pool_spikes_pre_inh = np.zeros((self.output_size/self.pool_kernel_size,self.output_size/self.pool_kernel_size,\
                                     self.output_channels))
        pool_spikes_post_inh = np.zeros((self.output_size/self.pool_kernel_size,self.output_size/self.pool_kernel_size,\
                                     self.output_channels))
        nofImages = 0
        for epoch in range(self.epochs):
            bar = progressbar.ProgressBar(maxval=self.sim_time/self.tsteps,widgets=[progressbar.Bar('=', '[', ']'), ' ', \
                                                                    progressbar.Percentage()])
            bar.start()
            for t in range(self.sim_time):
                im_num = t/self.tsteps

                if(im_num%(self.homeostasis_const)==0):
                    self.updates_per_map_imgs = [0]*self.output_channels
                    
                if(t%self.tsteps==0):
                    nofImages+=1
                    #print(nofImages)
                    if(self.debug):
                        ip_img +=self.input[:,:,1,(im_num-1)*self.tsteps:(im_num)*self.tsteps].sum(axis=2)                  
                        plt.imshow(ip_img,interpolation='none')
                        plt.title('Input Image',fontsize=font_size)
                        cplt = plt.colorbar()
                        cplt.ax.tick_params(labelsize=label_size)
                        plt.tick_params(labelsize=label_size)
                        plt.show()

                        plt.imshow(np.sum(spikes_pre_inh,axis=2),interpolation='none')
                        plt.title('Spikes in Conv1(L2) before inhibition',fontsize=font_size)
                        cplt = plt.colorbar()
                        cplt.ax.tick_params(labelsize=label_size)
                        plt.tick_params(labelsize=label_size)
                        plt.show()

                        plt.imshow(np.sum(spikes_post_inh,axis=2),interpolation='none')
                        plt.title('Spikes in Conv1(L2) after inhibition',fontsize=font_size)
                        cplt = plt.colorbar()
                        cplt.ax.tick_params(labelsize=label_size)
                        plt.tick_params(labelsize=label_size)
                        plt.show()      

                        if(self.train):
                            maps_spiked = np.where(spikes_post_stdp>=1)[-1]
                            pos_spiked = np.where(spikes_post_stdp>=1)[0:2]
                            plt.imshow(np.sum(spikes_post_stdp,axis=(2)),interpolation='none')
                            plt.title('Spikes in Conv1(L2) after stdp competition',fontsize=font_size)
                            plt.suptitle('X:{},Y:{},Z:{}'.format(pos_spiked[0],pos_spiked[1],maps_spiked),fontsize=font_size)
                            cplt = plt.colorbar()
                            cplt.ax.tick_params(labelsize=label_size)
                            plt.tick_params(labelsize=label_size)
                            plt.show()

                        plt.imshow(np.sum(pool_spikes_pre_inh,axis=2),interpolation='none')
                        plt.title('Spikes in Pool before inhibition',fontsize=font_size)
                        cplt = plt.colorbar()
                        cplt.ax.tick_params(labelsize=label_size)
                        plt.tick_params(labelsize=label_size)
                        plt.show()

                        plt.imshow(np.sum(pool_spikes_post_inh,axis=2),interpolation='none')
                        plt.title('Spikes in Pool after inhibition',fontsize=font_size)
                        cplt = plt.colorbar()
                        cplt.ax.tick_params(labelsize=label_size)
                        plt.tick_params(labelsize=label_size)
                        plt.show()
                    ip_img = np.zeros((self.size,self.size))
                    spikes_pre_inh = np.zeros((self.output_size,self.output_size,self.output_channels))
                    spikes_post_inh = np.zeros((self.output_size,self.output_size,self.output_channels))
                    spikes_post_stdp = np.zeros((self.output_size,self.output_size,self.output_channels))
                    pool_spikes_pre_inh = np.zeros((self.output_size/self.pool_kernel_size,self.output_size/self.pool_kernel_size,\
                                                 self.output_channels))
                    pool_spikes_post_inh = np.zeros((self.output_size/self.pool_kernel_size,self.output_size/self.pool_kernel_size,\
                                                 self.output_channels))
                    self.pots = np.zeros((self.output_size, self.output_size, self.output_channels),dtype = dtype)
                    self.pots_ = np.zeros((self.output_size, self.output_size, self.output_channels),dtype = dtype)
                    self.StdpPerNeuronAllowed = np.ones((self.output_size,self.output_size,self.output_channels),dtype = dtype)
                    self.SpikesPerNeuronAllowed = np.ones((self.output_size,self.output_size),dtype = dtype)
                    self.pool_pots = np.zeros((self.output_size/self.pool_kernel_size, self.output_size/self.pool_kernel_size, \
                                               self.output_channels),dtype = dtype)
                    self.pool_accumulated_voltages = np.zeros(self.pool_pots.shape)
                    self.pool_SpikesPerNeuronAllowed = np.ones((self.output_size/self.pool_kernel_size,\
                                                                self.output_size/self.pool_kernel_size),dtype = dtype)
                    self.conv_spikes = np.zeros((self.output_size,self.output_size,self.output_channels),dtype=dtype)
                    self.pool_spikes = np.zeros((self.output_size/self.pool_kernel_size,self.output_size/self.pool_kernel_size,\
                                                 self.output_channels),dtype)
                    
                self.pots+=self.convolution(self.input[:,:,:,t])

                #USE THIS NEXT 3 LINES IF YOU WANT CONVOLUTION USING SCIPY
                ##Weights = self.weights[0,0,:,:,:].reshape(self.conv_kernel_size,self.conv_kernel_size,\
                ##                                         self.input_channels,self.output_channels)
                ##self.pots+=self.scipy_convolution(self.input[:,:,0,t],Weights[:,:,0,:])
                ##self.pots+=self.scipy_convolution(self.input[:,:,1,t],Weights[:,:,1,:])
                self.spiked_neurons= np.where(self.pots>=self.threshold)
                self.conv_spikes[self.spiked_neurons[0],self.spiked_neurons[1],self.spiked_neurons[2]]=1
                spikes_pre_inh+=self.conv_spikes
                

                if(self.conv_lateral_inh):
                    self.conv_spikes,self.SpikesPerNeuronAllowed=self.lateral_inhibition(self.pots,self.conv_spikes,self.spiked_neurons,\
                                                                  self.SpikesPerNeuronAllowed)
                spikes_post_inh+=self.conv_spikes
                if(self.train):
                    stdp_spikes_per_t = 0
                    self.pots_ = self.pots*self.StdpPerNeuronAllowed*self.conv_spikes
                    if(self.pots_.sum() != 0):
                        for i in range(self.output_channels):
                            if(self.few_spikes):
                                homeosupp = np.ones((self.output_size,self.output_size,self.output_channels))
                                if(self.updates_per_map_imgs[i]>self.homeostasis_const/self.factor):
                                    homeosupp[:,:,i] = 0
                                    self.pots_ *= homeosupp
                                    self.weights[:,:,:,:,i] -= self.homeo_supp_coeff*self.weights[:,:,:,:,i]*(1-self.weights[:,:,:,:,i])
                        self.final_stdp_ids = self.stdp_competition(self.pots_)
                        for i in range(self.output_channels):
                            
                            if(len(self.final_stdp_ids[str(i)])!=0):
                                map_num=i
                                post_neur_id = self.final_stdp_ids[str(i)][0]
                                x_index = post_neur_id/self.output_size
                                y_index = post_neur_id%self.output_size
                                pre_neurons = self.connectivity[post_neur_id]
                                #self.update_weights(pre_neurons,self.weights[x_index,y_index,:,:,map_num],map_num,\
                                #                    self.A_plus, self.A_minus,im_num,t)
                                self.update_weights(pre_neurons,self.weights[0,0,:,:,map_num],map_num,\
                                                    self.A_plus, self.A_minus,im_num,t)
                                stdp_spikes_per_t += 1
                                self.updates_per_map[map_num] += 1
                                self.updates_per_map_imgs[map_num] += 1
                                suppression = supp_region.genregion(post_neur_id, self.output_size, self.inh_reg)
                                smol_reg = suppression.GenRegion()
                                self.StdpPerNeuronAllowed[smol_reg/self.output_size,smol_reg%self.output_size,:] = 0
                                self.StdpPerNeuronAllowed[:,:,map_num] = 0
                                spikes_post_stdp[post_neur_id/self.output_size,post_neur_id%self.output_size,map_num] += 1
                    
                    self.stdp_spikes.append(stdp_spikes_per_t)            
                    if(t%(self.tsteps*self.sample_interval-1)==0):
                        self.record_weights()
                    #if(t%(self.lr_inc_rate*self.tsteps-1)==0):
                    #print((nofImages-1)*self.tsteps+t%self.tsteps)
                    #if(((nofImages-1)*(self.tsteps)+t-1) % (self.lr_inc_rate*self.tsteps)==0):
                    if(((nofImages-1)*(self.tsteps)+t%self.tsteps) % ((self.lr_inc_rate)*self.tsteps-1)==0):
                        print('Images trained so far:{}'.format(nofImages))
                        self.A_plus = min(self.A_plus*2,0.15)
                        self.A_minus = 0.75*self.A_plus
                        print('current A_plus:{}, A_minus:{}, tstep:{}, epoch:{}'.format(self.A_plus, self.A_minus, t,\
                        epoch))

                if(not self.train):
                    if(self.pool_spike_accum):
                        self.pool_pots += self.pooling(self.conv_spikes*self.pots)
                    else:
                        self.pool_pots = self.pooling(self.conv_spikes*self.pots)
                    self.pool_spikes = (self.pool_pots>=self.threshold) ##pool neurons don't integrate no summation
                    pool_spikes_pre_inh += self.pool_spikes
                    if(self.pool_lateral_inh):
                        self.pool_spiked_neurons = np.where(self.pool_pots >= self.threshold)
                        self.pool_spikes, self.pool_SpikesPerNeuronAllowed= \
        self.lateral_inhibition(self.pool_pots,self.pool_spikes,self.pool_spiked_neurons,self.pool_SpikesPerNeuronAllowed)                

                    pool_spikes_post_inh += self.pool_spikes            
                    if(self.save_pool_spike_tensor):
                        self.pool_spike_tensor[:,:,:,t] = self.pool_spikes
                    if(self.save_pool_features):
                        #self.pool_pot_features[im_num] += theano.shared(self.pool_pots*self.pool_spikes).dimshuffle(2,0,1).eval().flatten()
                        ##self.pool_pots is reshaped to Z,X,Y and flattened so that neurons belonging to a map a grouped together.
                        self.pool_pot_features[:,:,:,im_num] += self.pool_spikes*self.pool_pots
                        self.pool_spike_features[:,:,:,im_num] += self.pool_spikes
                       
                bar.update(im_num)
                
            print('Epoch:{} is done!'.format(epoch))
        return

    def lateral_inhibition(self,pots,conv_spikes,spiked_neurons,SpikesPerNeuronAllowed): 
        '''
        Adapted from Saeed Reza Kheradpishe and Timothee Masquelier's Matlab code
        '''
        vbn = np.where(SpikesPerNeuronAllowed==0)
        conv_spikes[vbn[0],vbn[1],:]=0 #if a neuron in a position has spiked b4 don't let it spike 
        high_volts=np.zeros(pots.shape)
        idx = pots.argmax(axis=-1)
        high_volts[np.arange(pots.shape[0])[:,None],np.arange(pots.shape[1]),idx] = 1
        bvc = np.where(conv_spikes*high_volts==1)
        conv_spikes[bvc[0],bvc[1],bvc[2]]=1 #aross layers, only neurons with high pot get to keep spikes
        bvc1 = np.where(conv_spikes*high_volts!=1)
        conv_spikes[bvc1[0],bvc1[1],bvc1[2]]=0 #all other neurons that spiked with low pots dont get to spike
        spiked_neurons = np.where(conv_spikes==1)
        SpikesPerNeuronAllowed[spiked_neurons[0],spiked_neurons[1]]=0

        return conv_spikes, SpikesPerNeuronAllowed


    def stdp_competition(self,pots_):
        '''
        Adapted from Saeed Reza Kheradpishe and Timothee Masquelier's Matlab code
        '''
        tentative_stdp_ids={}
        for i in range(self.output_channels):
            tentative_stdp_ids[str(i)]=[]
        mxv = pots_.max(axis=2)
        mxi = pots_.argmax(axis=2)
        maxind1 = np.ones((self.output_channels,))*-1
        maxind2 = np.ones((self.output_channels,))*-1
        maxval = np.ones((self.output_channels,))*-1
        while(mxv.sum()!=0):
            maximum = mxv.max(axis=1)
            index = mxv.argmax(axis=1)
            index1 = maximum.argmax()
            index2 = index[index1]
            maxval[mxi[index1,index2]]=mxv[index1,index2]
            maxind1[mxi[index1,index2]]=index1
            maxind2[mxi[index1,index2]]=index2
            lkj = np.where(mxi==mxi[index1,index2])
            mxv[lkj[0],lkj[1]]=0;
            mxv[max(index1-self.offset,0):min(index1+self.offset+1,self.output_size),max(index2-self.offset,0):min(index2+\
                                                                                self.offset+1,self.output_size)]=0

        for i in range(maxind1.size):
            if(maxind1[i]!=-1 and maxind2[i]!=-1):
                tentative_stdp_ids[str(i)].append(int(self.output_size*maxind1[i]+maxind2[i]))

        return tentative_stdp_ids


    def update_weights(self,pre_neurons,weights2updt_,map_num,A_plus, A_minus,im_num,t):
        weights2updt = np.zeros((self.conv_kernel_size,self.conv_kernel_size,self.input_channels))
        for i in range(self.input_channels):
            weights2updt[:,:,i] = weights2updt_[:,i].reshape((self.conv_kernel_size,self.conv_kernel_size))
        rows = pre_neurons/self.size
        cols = pre_neurons%self.size
        ips = self.input[rows[0]:rows[-1]+1,cols[0]:cols[-1]+1,:,self.tsteps*(im_num):t+1].sum(axis=3).astype(dtype)#(np.float64)
        ips[np.where(ips>1)]=1
        weights2updt+=(ips*A_minus*weights2updt*(1-weights2updt))+(ips*A_plus*weights2updt*(1-weights2updt))-\
                       (A_minus*weights2updt*(1-weights2updt))
        #weights2updt[np.where(weights2updt>0.99)]=0.99
        #weights2updt[np.where(weights2updt<0.01)]=0.01
        weights2updt = weights2updt.reshape((self.conv_kernel_size**2,self.input_channels))
        #weights2updt = np.tile(weights2updt,(self.output_size,self.output_size,1,1))
        #self.weights[:,:,:,:,map_num]=weights2updt
        self.weights[0,0,:,:,map_num]=weights2updt
        
        return


    def record_weights(self):
        record_weights = deepcopy(self.weights[0,0,:,:,:])
        self.evol_weights.append(record_weights.reshape(self.conv_kernel_size,self.conv_kernel_size,self.input_channels,-1))
        

    def animation(self,features, sample_interval,intervals,plotx,ploty,layer_num=[2,3,4],filter_sizes=[5,2,5],\
                             filter_strides=[1,2,1],nof_filters=[30,30,100],types=['conv','pool','conv'],currLayer=4,\
                             figsize=(10,10), font1=14, font2=20):
        if(self.train):
            if(any(isinstance(el, list) for el in features)):
                pass
            else:
                print('Exiting!! features, should be a list of list')
                sys.exit()
            if(len(features)==1): ## feature visualization for the first layer
                features = features[0]
                #features = [items.reshape(self.conv_kernel_size,self.conv_kernel_size,self.input_channels,self.output_channels) for items\
                #            in self.evol_weights]
        # make sure that the shape of the features is [5x5x2x30]*#samples
        #########################################################################################################
                plot_x=plotx; plot_y=ploty
                fig, axes = plt.subplots(plot_x,plot_y , figsize=figsize,
                                 subplot_kw={'xticks': [], 'yticks': []})
                fig.subplots_adjust(left=0.12, bottom=0.0, right=0.89, top=0.9, wspace=0.1, hspace=0.21)
                axes_list = []
                for i, sub in enumerate(axes.flatten()):
                    X = np.zeros((self.conv_kernel_size,self.conv_kernel_size,3))
                    X[:,:,0]=features[0][:,:,1,i]
                    X[:,:,1]=features[0][:,:,0,i]
                    sub.set_title('Map'+str(i+1),fontsize=font1)
                    axes_list.append(sub.imshow(X,interpolation='none'))

        #set up the skeleton
        #########################################################################################################################
                def run(it):
                    for nd, subaxes in enumerate(axes_list):    
                        X = np.zeros((self.conv_kernel_size,self.conv_kernel_size,3))
                        X[:,:,0]=features[it][:,:,1,nd]
                        X[:,:,1]=features[it][:,:,0,nd]
                        self.final_weights[:,:,:,nd]=X
                        subaxes.set_data(X)
                    return axes_list
                #### function to update the (skeleton) animation 
        #########################################################################################################################

                ani=animation.FuncAnimation(fig, run, frames=range(1,len(features)),blit=True)
                #plt.show()
            else:
                first_layers_features =[[[items[0][-1]]] for items in features[0:len(features)-1]]
                final_features = []
                for i in range(len(features[-1][0])):
                    features_it = []
                    features_it.extend(first_layers_features)
                    features_it.append([[features[-1][0][i]]])
                    #print([items[0].shape for items in features_it])
                    self.feature_visualization(features_it,sample_interval,intervals,plotx,ploty,\
                                    layer_num,filter_sizes,filter_strides,nof_filters,types,currLayer,show=False)
                    final_features.append(self.final_features)
                #print(len(final_features),[items.shape for items in final_features])
                
                plot_x=plotx; plot_y=ploty
                fig, axes = plt.subplots(plot_x,plot_y , figsize=figsize,
                                 subplot_kw={'xticks': [], 'yticks': []})
                fig.subplots_adjust(left=0.03, bottom=0.0, right=0.99, top=0.9, wspace=0.27, hspace=0.21)
                axes_list = []
                self.feature_visualization(features,sample_interval,intervals,plotx,ploty,\
                                            layer_num,filter_sizes,filter_strides,nof_filters, types,currLayer,show=False)
                for i, sub in enumerate(axes.flatten()):
                    sub.set_title('Map'+str(i+1),fontsize=font1)
                    axes_list.append(sub.imshow(self.final_features[:,:,:,i],interpolation='none'))
                    ###TODO: Fix this
                
                def run(it):
                    for nd, subaxes in enumerate(axes_list):    
                        subaxes.set_data(final_features[it][:,:,:,nd])
                    return axes_list

                ani=animation.FuncAnimation(fig, run, frames=range(len(features[-1][0])),repeat=not False,blit=True)
                #ani=animation.FuncAnimation(fig, run, frames=len(features[-1])-1,repeat=False)
                #plt.show()

            plt.suptitle('Final filters of Layer:{}'.format(currLayer),fontsize=font2)
            return ani, fig
        else:
            print('THIS METHOD IS USED WHEN train in __init__ IS SET TO True') 
            



    def feature_visualization(self, features, sample_interval,intervals,plotx=6,ploty=5,layer_num=[2,3,4],filter_sizes=[5,2,5],\
                             filter_strides=[1,2,1],nof_filters=[30,30,100],types=['conv','pool','conv'],currLayer=4,\
                             show=True, figsize=(10,10), font1=12, font2=18):
        if(self.train):
            if(any(isinstance(el, list) for el in features)):
                pass
            else:
                print('Exiting!! features, should be a list of list')
                sys.exit()

            samples = [0+sample_interval*i for i in range(0,intervals+1)]  
            if(len(features)==1): ## feature visualization for the first layer
                features = features[0]
                self.final_features = features
                fig, axes = plt.subplots(len(features), self.output_channels,figsize=figsize,
                             subplot_kw={'xticks': [], 'yticks': []})
                #fig.subplots_adjust(left=0.12,bottom=0.04,right=0.89,top=1.0,hspace=0.42, wspace=0.08)
                for i, row in enumerate(axes):
                    for j, cell in enumerate(row):
                        x = features[i][:,:,0,j]
                        x_ = features[i][:,:,1,j]
                        #x = x.reshape(self.conv_kernel_size,self.conv_kernel_size)
                        #x_=x_.reshape(self.conv_kernel_size,self.conv_kernel_size)
                        X = np.zeros((self.conv_kernel_size,self.conv_kernel_size,3))
                        X[:,:,0]=x_
                        X[:,:,1]=x
                        cell.imshow(X,interpolation='none')
                        if i == len(axes) - 1:
                            cell.set_xlabel("Map: {0:d}".format(j + 1),rotation='vertical',fontsize=font1)
                        if j == 0:
                            cell.set_ylabel("{0:d}         ".format(samples[i]),rotation=0,fontsize=font1)

                #plt.suptitle('Evolved Filters for {} Images and A_plus=0.004,A_minus=0.003'.\
                #             format(len(features)*sample_interval),fontsize=30)
                fig.text(0.04, 0.5, 'Evolved filters for every'+' '+str(sample_interval)+' '+'images',\
                 va='center', rotation='vertical',fontsize=font2)
                #plt.tight_layout()
                #plt.show()
            else:
                feature_sizes=[]
                layer_weights = [items[0][-1] for items in features]
                feature_size=0
                index = layer_num.index(currLayer)
                temp =filter_sizes[index]
                for index_ in range(index,0,-1):
                    feature_size = (temp*filter_strides[index_-1]+filter_sizes[index_-1])
                    feature_sizes.insert(0,feature_size)
                    temp = feature_size
                layer = layer_num[index]
                layer_weight_index = (layer/2)-1
                conv_features = layer_weights[layer_weight_index]
                for index_ in range(index,0,-1):
                    if(types[index_-1]=='pool'):
                        pool_features = np.zeros((feature_sizes[index_-1],\
                            feature_sizes[index_-1],nof_filters[index_-1],nof_filters[index]))
                        for axis_ in range(nof_filters[index]):
                            a_pool_feature = np.zeros((feature_sizes[index_-1],\
                                           feature_sizes[index_-1],nof_filters[index_-1]))
                            a_conv_feature = conv_features[:,:,:,axis_]
                            locs = np.where(a_conv_feature>0.3)
                            pool_locs = [filter_strides[index_-1]*items for items in locs[0:2]]
                            pool_locs.append(locs[-1])
                            pool_locs = tuple(pool_locs)
                            a_pool_feature[pool_locs]=a_conv_feature[locs]
                            pool_features[:,:,:,axis_]=a_pool_feature
                    elif(types[index_-1]=='conv'):
                        if(index_==1):
                            final_features = np.zeros((feature_sizes[index_-1],\
                                               feature_sizes[index_-1],3,nof_filters[index]))
                            for axis_ in range(nof_filters[index]):
                                for iii in range(pool_features[:,:,:,axis_].shape[0]):
                                    for jjj in range(pool_features[:,:,:,axis_].shape[1]):
                                        mxv=pool_features[iii,jjj,:,axis_].max()
                                        mxi=pool_features[iii,jjj,:,axis_].argmax()
                                        if(mxv>0.1):
                                            strd=filter_strides[index_-1]
                                            szs=filter_sizes[index_-1]
                                            final_features[(iii-1)*strd+1:(iii)*strd+szs,\
                    (jjj-1)*strd+1:(jjj)*1+szs,0,axis_]+=pool_features[iii,jjj,mxi,axis_]*layer_weights[index_-1][:,:,1,mxi]
                                            final_features[(iii-1)*strd+1:(iii)*strd+szs,\
                    (jjj-1)*strd+1:(jjj)*1+szs,1,axis_]+=pool_features[iii,jjj,mxi,axis_]*layer_weights[index_-1][:,:,0,mxi]
                        else:
                            conv_features = np.zeros((feature_sizes[index_-1],\
                                               feature_sizes[index_-1],nof_filters[index_-2],nof_filters[index_+1]))
                            for axis_ in range(nof_filters[index]):
                                for iii in range(pool_features[:,:,:,axis_].shape[0]):
                                    for jjj in range(pool_features[:,:,:,axis_].shape[1]):
                                        mxv=pool_features[iii,jjj,:,axis_].max()
                                        mxi=pool_features[iii,jjj,:,axis_].argmax()
                                        if(mxv>0.3):
                                            strd=filter_strides[index_-1]
                                            szs=filter_sizes[index_-1]
                                            layer = layer_num[index_]
                                            layer_weight_index = (layer/2)-1
                                            conv_features[(iii-1)*strd+1:(iii)*strd+szs,\
                    (jjj-1)*strd+1:(jjj)*1+szs,:,axis_]+=pool_features[iii,jjj,mxi,axis_]*layer_weights[layer_weight_index][:,:,:,mxi]

                self.final_features = final_features
                if(show):
                    fig, axes = plt.subplots(plotx, ploty,figsize=figsize,
                                 subplot_kw={'xticks': [], 'yticks': []})
                    fig.subplots_adjust(left=0.03, bottom=0.0, right=0.99, top=0.9, wspace=0.27, hspace=0.21)      
                    axes = axes.flat
                    for i in range(len(axes)):
                        axes[i].imshow(final_features[:,:,:,i],interpolation='none')
                        axes[i].set_title('Map'+str(i+1),fontsize=16)
                    #plt.show()
                
                    return fig
            return fig
        else:
            print('THIS METHOD IS USED WHEN train in __init__ == True and save_pool_spike_tensor==True') 


    def feature_convergence(self, features, sample_interval, font_size=38):
        if(self.train):
            keys = ['Conv'+str(i+1) for i in range(len(features))]
            convergences = [[] for items in features]
            differences = [[] for items in features]
            nofWeights = [items[0].size for items in features]
            for layer in range(len(features)):
                for sample in range(1,len(features[layer])):
                    differences[layer].append(((features[layer][sample-1]-features[layer][sample])**2).sum()/nofWeights[layer])
                    convergences[layer].append((features[layer][sample]*(1-features[layer][sample])).sum()/nofWeights[layer])
            convdict = {k:v for k,v in zip(keys,convergences)}
            diffdict = {k:v for k,v in zip(keys, differences)}

            df = pd.DataFrame.from_dict(convdict,orient='index')
            df = df.transpose()
            df.plot(style='.-',title='Plot of weight convergence')
            plt.ylabel('$\\frac{\\Sigma(Weights*(1-Weights))}{No.\\ Of\\ Weights}$',fontsize=font_size)
            #df.plot(style='.-',title='Plot of weight convergence',logy= True)
            #plt.ylabel('$\\log(\\frac{Weights*(1-Weights)}{No.\\ Of\\ Weights})$')
            plt.xlabel('Sample Number, S.I='+str(sample_interval)+' Images')
            plt.show()

            df = pd.DataFrame.from_dict(diffdict,orient='index')
            df = df.transpose()
            df.plot(style='.-',title='Plot of normalized temporal difference of weights')
            plt.ylabel('$\\frac{\\Sigma(Weights[t-1]-Weights[t])^{2}}{No.\\ Of\\ Weights}$',fontsize=font_size)
            plt.xlabel('Sample Number, S.I='+str(sample_interval)+' Images')
            plt.show()
        else:
            print('THIS METHOD IS USED WHEN train in __init__ IS SET TO True') 

        return


    def spike_statistics(self, font_size=38):
        if(self.train):           
            plt.bar(range(1,len(self.updates_per_map)+1),self.updates_per_map)
            plt.xlabel('Maps',fontsize=font_size)
            plt.ylabel('Number of weight updates',fontsize=font_size)
            plt.xlim(1,len(self.updates_per_map)+1)
            plt.ylim(0,max(self.updates_per_map))
            #plt.tick_params(labelsize=28)
            plt.title('Bar chart of number of weight updates')
            plt.show()

            fig, ax = plt.subplots()
            temporal_spikes = self.stdp_spikes
            temporal_spikes=np.add.reduceat(temporal_spikes,range(0,self.epochs*self.sim_time,self.tsteps))
            ax.scatter(range(len(temporal_spikes)),temporal_spikes,marker='*',s=6)
            ax.set_xlabel('#Image',fontsize=font_size)
            ax.set_ylabel('No.Of STDP Spikes',fontsize=font_size)
            ax.set_xlim(0,len(temporal_spikes))
            ax.set_ylim(0,max(temporal_spikes)+3)
            spikes_ms = np.mean(temporal_spikes).round(decimals=3)
            ax.text(2000,4.5,'Mean='+str(spikes_ms)+" spikes/image",fontsize=font_size,bbox=dict(facecolor='none', edgecolor='blue',pad=2))
            sub_axis = inset_axes(ax,width='25%',height=2.5,loc=1)
            x1,x2,y1,y2=1000,1200,0,spikes_ms+2
            #print(len(temporal_spikes))
            sub_axis.scatter(range(x1,x2),temporal_spikes[x1:x2], marker='*',s=6)
            sub_axis.tick_params(labelsize=font_size/2)
            sub_axis.set_xlim(x1,x2)
            sub_axis.set_ylim(y1,y2)
            sub_axis.ticklabel_format(scilimits=(0,0),axis='x')
            mark_inset(ax,sub_axis,loc1=2,loc2=4,fc='none',ec='0.5')
            plt.show()
            return fig

        else:
            print('THIS METHOD IS USED WHEN train in __init__ IS SET TO True') 



    def spikes_per_map_per_class(self,plot_x,plot_y,class_labels,pool_output_data,labels_map,labelsize, view_maps,\
    final_weights, figsize=(16,8)):
        '''
        offset: since spikes per map per label for all feature maps can't be plotted because of space issues,
                we selectively plot for some feature maps
        labels_map: A dictionary of label_number:label 
        pool_output_data: spike tensor collected at a pooling layer
        final_weights: final evolved weights
        view_maps: Map numbers to be examined
        '''
        if(not self.train and self.save_pool_spike_tensor):
            rows = plot_y
            cols = plot_x
            fig, axes = plt.subplots(rows, cols,figsize=figsize)
            fig.subplots_adjust(hspace=0.20, wspace=0.07,top=0.94, right=1.00, bottom=0.11, left=0.03)
            ax = axes.flat
            a_sum = 0
            select_index=[]
            for i in range(len(ax)):
                spikes_per_digit={}
                for dig in range(max(class_labels)+1):
                    spikes_per_digit[labels_map[dig]]=0

                for t in range(pool_output_data.shape[-1]):
                    #print(class_labels[t])
                    spikes_per_digit[labels_map[int(class_labels[t/self.tsteps])]]+=pool_output_data[:,:,view_maps[i]-1,\
                    t].sum()

                for dig in range(max(class_labels)+1):
                    a_sum+=spikes_per_digit[labels_map[dig]]
                bars = ax[i].bar(range(len(spikes_per_digit)), spikes_per_digit.values(),tick_label=spikes_per_digit.keys())
                reds = sorted(range(len(spikes_per_digit.values())), key=lambda po: spikes_per_digit.values()[po])[-5:]
                for red in reds:
                    bars[red].set_color('k')
                #reds.sort()
                select_index.append(spikes_per_digit.values())
                ax[i].set_title('Map'+str(view_maps[i])+' ,'+''+'Dominant classes:'+str([spikes_per_digit.keys()[red] \
                for red in reds]),fontsize=font_size/2)
                #ax[i].set_ylim(0,2*(spikes_per_digit[labels_map[reds[-1]]]))
                ax[i].tick_params(labelrotation=45)
                ax[i].tick_params(axis='both', which='major', labelsize=labelsize)
                small_axes = inset_axes(ax[i],width="20%",height="20%",loc=1)
                small_axes.tick_params(axis='both', which='major', labelsize=8)
                if(type(final_weights) == None):
                    print('use this method after using feature_visualization method!')
                x = np.zeros((final_weights.shape[0],final_weights.shape[1],3))
                ##note that final_weights should receive final reconstruced features (i.e self.final_features) which has
                ## channels flipped for viz. RGB, red is for OFF and green is for ON. So, we don't need to flip again
                x[:,:,0]=final_weights[:,:,0,view_maps[i]-1] 
                x[:,:,1]=final_weights[:,:,1,view_maps[i]-1]
                small_axes.imshow(x,interpolation='None')
                #print(sum(spikes_per_digit.values()))
                #print(spikes_per_digit.values())

            matplotlib.rcParams.update({'font.size': font_size})
            #plt.suptitle('Spike profile of {} feature maps that are selective to {} different classes'.\
            #             format(str(pool_output_data.shape[2]),str(max(class_labels)+1)))
            #plt.show()
            print('TOTAL NUMBER OF SPIKES IN POOL1 TIME TENSOR IN PLOTS:{}'.format(a_sum))
            print('TOTAL NUMBER OF SPIKES IN POOL1 TIME TENSOR:{}'.format(pool_output_data.sum()))
            return fig
        else:
            print('THIS METHOD IS USED WHEN train in __init__ IS SET TO False') 

    def save_weights(self):
        picklefile='weights'+'.pkl'
        output = open(picklefile,'wb')
        pickle.dump(self.evol_weights,output)
        output.close()
        print('Saved on weights to:{}'.format(picklefile))

    
    def make_feature_vecs(self,tensor):
        ### above lines so that when reshaped, all the neurons of a map stay together.(by default numpy flattens around
        ## first axes, if array is of shape (t,3,3,200) and if we want to reshape it as (t,1800) it is reshaped  as
        ##(t,3x600)  grouping axes 2,3 (3,200)
        ## together instead of grouping 1,2 (3,3) reshaping the tensor as (t,200,3,3) will help with that and group axis
        ##with dims 3,3 together.
        #return theano.shared(np.add.reduceat(tensor,range(0,self.sim_time,self.tsteps),axis=3)).\
        #                        dimshuffle(3,2,0,1).eval().reshape(int(self.sim_time/self.tsteps),-1)
        ## use the above line if you want pots/spikes at every tstep
        if(self.save_pool_features):
            temp = theano.shared(tensor).dimshuffle(3,2,0,1).eval().reshape(int(self.sim_time/self.tsteps),-1)

        else:
            print(' you did not save pool features, set save_pool_features to True and then try again')
            sys.exit()

        return temp


    def faster_max_feature_vecs(self,step_nofImages, nof_conv_layers = 2):
        if(nof_conv_layers > 1):
            ############# setup the input spikes(previous pool layer) and kernels
            prev_pool_spikes = theano.shared(self.input) ## self.sizexself.sizexself.input_channelsxself.sim_time
            prev_pool_spikes = prev_pool_spikes.dimshuffle(3,0,1,2)
            prev_pool_spikes = prev_pool_spikes.eval()
            prev_pool_spikes = prev_pool_spikes[np.newaxis,:,:,:,:]
            ## above reshaping because tensorflow takes [batch, in_depth, in_height, in_width, in_channels] 
            ## see https://www.tensorflow.org/api_docs/python/tf/nn/conv3d
            Weights = self.weights[0,0,:,:,:]
            Weights = Weights.reshape(1,self.conv_kernel_size,self.conv_kernel_size,self.input_channels,self.output_channels)
            ##above reshaping is required because tensorflow takes [filter_depth, filter_height, filter_width, in_channels,
            #out_channels] as kernel dimensions for a 3d convolution.

            ############setup the convolution
            feature_vecs = np.zeros((self.nofImages,self.output_channels),dtype = np.float32)
            iters = self.nofImages/step_nofImages
            intervals = [i*step_nofImages for i in range(iters+1)]
            if(intervals[-1] != self.nofImages):
                intervals[-1] += self.nofImages-intervals[-1]
            def video_tf_convolution(Images, Weights, inter):
                input_5d = Images
                kernel_5d = Weights
                output_4d = sess.run(inter,feed_dict ={image_placeholder:input_5d,kernel_placeholder:kernel_5d})
                return output_4d
            bar = progressbar.ProgressBar(maxval=iters,widgets=[progressbar.Bar('=', '[', ']'), ' ', \
                                                                    progressbar.Percentage()])
            bar.start()
            for i in range(iters):
                step_nofImages = intervals[i+1] - intervals[i]
                image_placeholder = tf.placeholder(tf.float32,shape=[1,self.tsteps*step_nofImages,\
                    self.size,self.size,self.input_channels])
                kernel_placeholder = tf.placeholder(tf.float32, shape=[1,\
                    self.conv_kernel_size,self.conv_kernel_size,self.input_channels,self.output_channels])
                strides_5d = [1, 1, 1, 1, 1]
                inter = tf.squeeze(tf.nn.conv3d(image_placeholder, kernel_placeholder, \
                    strides = strides_5d, padding = 'VALID'))
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                results_4d = video_tf_convolution(prev_pool_spikes[:,intervals[i]*self.tsteps:intervals[i+1]*self.tsteps,:,:,:],\
                Weights, inter)
                results_2d = results_4d.max(axis=(1,2)) ##take max value in a feature map for every tstep
                ##**results_4d = tf.math.reduce_max(4d_result,axis=(1,2)).eval() ##take max value in a feature map for every tstep
                
                img_feature_vec = np.add.reduceat(results_2d,range(0,self.tsteps*step_nofImages,self.tsteps),axis = 0)
                
                ##**results_4d = np.add.reduceat(results_4d,range(0,self.tsteps*step_nofImages,self.tsteps),axis = 0)
                ##**img_feature_vec = results_4d.max(axis=(1,2)) 

                
                
                feature_vecs[intervals[i]:intervals[i+1],:] = img_feature_vec
                bar.update(i+1)        
        else:
            print('This method is usable only if the network has more than 1 conv layer!!')
            sys.exit()
        return feature_vecs

