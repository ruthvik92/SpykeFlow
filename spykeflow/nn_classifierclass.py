'''
classifierclass.py
'''
import keras
from keras.models import Sequential
from keras import backend
from keras.layers import Dense, Activation, Dropout, LeakyReLU
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from keras.layers.normalization import BatchNormalization
import numpy_fcn
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import regularizers
import sys, os, math
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras import backend
from keras.callbacks import LearningRateScheduler
from clr_callback import *
from keras.utils.vis_utils import plot_model
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
class Swish(Activation):
    '''
    https://github.com/keras-team/keras/issues/8716
    '''
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x, beta=1):
    '''
    https://www.bignerdranch.com/blog/implementing-swish-activation-function-in-keras/
    '''
    return (x*sigmoid(beta*x))
get_custom_objects().update({'swish':Swish(swish)})


class spkNeuron(Activation):
    '''
    https://github.com/keras-team/keras/issues/8716
    '''
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'spkNeuron'

def spkNeuron(x, theta=0):
    
    if(x>theta):
        return 1
    else:
        return 0

get_custom_objects().update({'spkNeuron':spkNeuron(spkNeuron)})
#np.random.seed(0)
##os.environ["CUDA_VISIBLE_DEVICES"]="-1"

## for rc params updates see
## https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
## https://www.programcreek.com/python/example/104483/matplotlib.rcParams.update
## matplotlib.rcParams.update({'font.size': font_size})

'''
Had to write a separate class because this was not implemented in the keras.callbacks.callbacks base astract class
'''
class LearningRateMonitor(Callback):
    def on_train_begin(self, logs={}):
        self.lrates = []

    def on_epoch_end(self, epochs, logs={}):
    
        lrate = float(backend.get_value(self.model.optimizer.lr))
        self.lrates.append(lrate)

backend.clear_session()

class Classifier(object):

    def __init__(self,train_data = None,val_frac = 0.10, test_data = None, lmbda = 0.1, network_structure= None, \
        activation_fns = None, batch_size = 10,epochs = 25,verbose=0,plots=True,optimizer='adam',loss_function =\
        'categorical_crossentropy',eta=0.5,patience=8,eta_decay_factor=1,eta_drop_type='gradual',epochs_drop=1,
        decay_rate = None,ip_lyr_drop_out=0,drop_out=0,leaky_relu=False,weight_init='glorot_uniform', batch_norm=False,
        leaky_alpha=0.1,bias_init=0,log_path=None):
        ##don't use loss_function = binary_crossentropy
        ### see https://stackoverflow.com/questions/42081257/keras-binary-crossentropy-vs-categorical-crossentropy-performance
        
        self.train_x, self.train_y = train_data
        self.val_frac = val_frac
        self.test_data = test_data
        self.test_x, self.test_y = test_data
        self.lmbda = lmbda
        self.c = 1.0/(self.lmbda+0.0000000001)
        self.network_structure = network_structure
        self.activation_fns = activation_fns
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.plots = plots
        self.keras_net = None
        self.numpy_net = None
        self.model = Sequential()
        self.history =  None
        self.patience = patience
        self.log_path = log_path
        self.eta_decay_factor = eta_decay_factor
        self.eta_drop_type = eta_drop_type
        self.epochs_drop = epochs_drop
        self.eta = eta
        self.decay_rate = self.eta / self.epochs
        self.best_val = 0
        self.weight_init = weight_init
        self.ip_lyr_drop_out = ip_lyr_drop_out
        self.drop_out = drop_out
        self.leaky_relu = leaky_relu
        self.batch_norm = batch_norm
        self.leaky_alpha = leaky_alpha
        self.bias_init = bias_init

    def gradual_eta_drop(self, epoch):
        '''
        Had to write as a method because keras.callbacks.callbacks has a class LearningRateScheduler that accepts
        methods different kinds of schedulers.
        '''
        init_eta = self.eta
        drop = self.eta_decay_factor
        epochs_drop = self.epochs_drop
        eta = init_eta/math.pow(drop,math.floor((1+epoch)/epochs_drop))
        return eta


    def keras_fcn_classifier(self):
        '''
            see: https://machinelearningmastery.com/check-point-deep-learning-models-keras/
            see: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
            see: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
        '''
        # build the network


        self.train_y = np_utils.to_categorical(np.array(self.train_y).reshape(-1,1),self.network_structure[-1])

        self.test_y = np_utils.to_categorical(np.array(self.test_y).reshape(-1,1),self.network_structure[-1])
        
        if(self.ip_lyr_drop_out != 0):
            #https://github.com/keras-team/keras/issues/96
            ## next two lines if you want to use dropout in input layer
            self.model.add(Dropout(self.ip_lyr_drop_out, input_shape = (self.network_structure[0],)))
            self.model.add(Dense(self.network_structure[1],kernel_initializer=self.weight_init,
            bias_initializer=keras.initializers.Constant(self.bias_init)))
            if(not self.leaky_relu):
                self.model.add(Activation(self.activation_fns[0]))
            else:
                self.model.add(LeakyReLU(alpha=0.1))

        else:
            ## next line if you want a simple input layer and an output layer (without dropout)
            self.model.add(Dense(self.network_structure[1],input_dim=self.network_structure[0],
            kernel_regularizer=regularizers.l1(self.lmbda),kernel_initializer=self.weight_init,
            bias_initializer=keras.initializers.Constant(self.bias_init))) ##input and hidden layers
            if(not self.leaky_relu):
                self.model.add(Activation(self.activation_fns[0]))
            else:
                self.model.add(LeakyReLU(alpha=0.1))
                

        for i in range(len(self.network_structure[2:])):
            if(self.drop_out != 0):
                self.model.add(Dropout(self.drop_out))
            self.model.add(Dense(self.network_structure[2+i],kernel_regularizer=regularizers.l1(self.lmbda),
            kernel_initializer=self.weight_init,bias_initializer=keras.initializers.Constant(self.bias_init)))
            if(self.batch_norm):
                self.model.add(BatchNormalization())
            #if(not self.leaky_relu):
            #    self.model.add(Activation(self.activation_fns[1+i]))
            #else:
            if(self.leaky_relu):
                if( i != len(self.network_structure[2:])-1):
                    self.model.add(LeakyReLU(alpha=self.leaky_alpha))
                else:
                    self.model.add(Activation(self.activation_fns[1+i]))
            else:
                self.model.add(Activation(self.activation_fns[1+i]))
        print(self.model.summary())
        # compile the model
        #plot_model(self.model,show_shapes=True,show_layer_names=True)
        if(self.optimizer == 'sgd'):
            momentum = 0.8
            self.optimizer = SGD(lr=self.eta, momentum = momentum, decay = self.decay_rate, nesterov=False)
        if(self.optimizer == 'adam'):
            self.optimizer = Adam(lr=self.eta, beta_1=0.9, beta_2=0.999, epsilon=None, 
            amsgrad=not False, clipnorm=1.0)
        ## compile model
        self.model.compile(loss=self.loss_function, optimizer = self.optimizer, metrics=['accuracy'])
        
        ## callbacks
        filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=self.verbose, save_best_only=True, mode='max')
        lrm = LearningRateMonitor()
        tb_callback = keras.callbacks.TensorBoard(log_dir=self.log_path,histogram_freq=2,write_graph=True,write_grads=True,batch_size=100)

        if(self.eta_drop_type == 'plateau'):
            rlrp = ReduceLROnPlateau(monitor='val_acc', factor= 1/self.eta_decay_factor, patience=self.patience, min_delta=1E-7)
            if(self.log_path!=None):
                callbacks_list = [checkpoint, rlrp, lrm, tb_callback]
            else:
                callbacks_list = [checkpoint, rlrp, lrm]

        
        elif(self.eta_drop_type == 'gradual'):
            reduce_eta_gradual = LearningRateScheduler(self.gradual_eta_drop)    
            if(self.log_path!=None):
                callbacks_list = [checkpoint, reduce_eta_gradual, lrm, tb_callback]
            else:
                callbacks_list = [checkpoint, reduce_eta_gradual, lrm]
        elif(self.eta_drop_type == 'cyclical'):
            #reduce_eta_cyclical = CyclicLR(mode='exp_range', gamma=0.99994, base_lr=0.005, max_lr=0.01, step_size=500)
            
            #clr = lambda x: 1/(5**(x*(0.0001/self.epochs)))
            #reduce_eta_cyclical = CyclicLR(scale_fn=clr,scale_mode='iterations',base_lr=0.005/2, max_lr=0.01/2, step_size=500)
            
            clr = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            reduce_eta_cyclical = CyclicLR(scale_fn=clr,scale_mode='cycle',base_lr=self.eta/2, max_lr=self.eta*2,
            step_size=500)
            if(self.log_path!=None): 
                callbacks_list = [checkpoint, reduce_eta_cyclical, lrm, tb_callback]
            else:
                callbacks_list = [checkpoint, reduce_eta_cyclical, lrm]


        # fit the network
        self.train_x = self.train_x.astype(np.float32)
        self.train_x = self.train_x / self.train_x.max()
        #self.train_x = self.train_x / float(self.train_x.max())
        #self.train_x = preprocessing.scale(self.train_x,axis=1)
        self.test_x = self.test_x.astype(np.float32)
        self.test_x = self.test_x / self.test_x.max()
        #self.test_x = self.test_x / float(self.test_x.max())
        #self.test_x = preprocessing.scale(self.test_x,axis=1)
        
        self.keras_net = self.model.fit(self.train_x, self.train_y, validation_split=self.val_frac, epochs=self.epochs, \
                batch_size=self.batch_size, verbose=self.verbose, callbacks=callbacks_list)
        self.history = self.keras_net.history
        self.best_val = max(self.history['val_acc'])
        if(self.plots):
            # plot accuracies
            if(self.eta_drop_type == 'cyclical'):
                self.get_plots([self.history['acc'],self.history['val_acc']],
                [self.history['loss'],self.history['val_loss']],[reduce_eta_cyclical.history['iterations'],
                reduce_eta_cyclical.history['lr']])
            else:
                self.get_plots([self.history['acc'],self.history['val_acc']],
                [self.history['loss'],self.history['val_loss']],lrm.lrates)
        
        # load best weights
        self.model.load_weights("weights.best.hdf5")
        # Compile model (required to make predictions)
        ##**self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])
        self.model.compile(loss=self.loss_function, optimizer = self.optimizer, metrics=['accuracy'])
        print('Testing the best model')
        scores = self.model.evaluate(self.test_x, self.test_y, verbose=self.verbose)
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

    

    def get_plots(self, acc_data, loss_data, etas):
        plt.plot(acc_data[0])
        plt.plot(acc_data[1])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        plt.show()
            
        plt.plot(loss_data[0])
        plt.plot(loss_data[1])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        plt.show()
        
        if(self.eta_drop_type == 'cyclical'):
            plt.xlabel('Iteration (#Mini batch)')
            plt.plot(etas[0],etas[1])
            plt.title('Learning Rate Monitor')
            plt.ylabel('Learning Rate')
        else:
            plt.xlabel('Epoch')
            plt.plot(etas)
            plt.title('Learning Rate Monitor')
            plt.ylabel('Learning Rate')
        plt.show()
    def numpy_fcn_classifier(self):
        train_x = self.train_x.astype(np.float32)
        train_x/=float(train_x.max())
        train_y = self.train_y
        train_x = preprocessing.scale(train_x,axis=0)

        test_x = self.test_x.astype(np.float32)
        test_x/=float(test_x.max())
        test_y = self.test_y
        test_x = preprocessing.scale(test_x,axis=0)

        training_inputs = [np.reshape(x, (self.network_structure[0], 1)) for x in train_x[int(len(train_x)*self.val_frac):]]
        training_results = [self.vectorized_result(y) for y in train_y[int(len(train_x)*self.val_frac):]]
        training_data = zip(training_inputs, training_results)

        validation_inputs = [np.reshape(x, (self.network_structure[0], 1)) for x in train_x[0:int(len(train_x)*self.val_frac)]]
        validation_data = zip(validation_inputs, train_y[0:int(len(train_x)*self.val_frac)])

        test_inputs = [np.reshape(x, (self.network_structure[0], 1)) for x in test_x]
        test_data = zip(test_inputs, test_y)
        
        ### build the network    
        self.numpy_net = numpy_fcn.Network(self.network_structure, cost=numpy_fcn.softmax) #QuadraticCost, CrossEntropyCost,softmax
        #### start gradient descent
        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, eta_history = \
        self.numpy_net.SGD(training_data,epochs=self.epochs, mini_batch_size= self.batch_size,
        eta=self.eta, lmbda=self.lmbda, evaluation_data=validation_data, monitor_evaluation_cost= True, 
        monitor_evaluation_accuracy= True, monitor_training_cost=True, monitor_training_accuracy=True,
        eta_decay_factor=self.eta_decay_factor, epochs_drop = self.epochs_drop)

        self.history ={'acc':training_accuracy, 'val_acc':evaluation_accuracy, 'loss':training_cost,
        'val_loss':evaluation_cost}
        if(self.plots):
            # plot accuracies
            self.get_plots([self.history['acc'],self.history['val_acc']],
            [self.history['loss'],self.history['val_loss']],eta_history)
        
        test_accuracy = self.numpy_net.accuracy(test_data)
        print('test accuracy at the end:{}'.format(test_accuracy/float(len(test_data))))
        print('best validation accuracy is:{}'.format(self.numpy_net.best_val))
        self.best_val = self.numpy_net.best_val
        self.numpy_net  = numpy_fcn.load('fcn_best_validation_network_rand_epochs_'+str(self.epochs)+'_lambda_'+str(self.lmbda))
        print('test accuracy for weights corresponding to the best validation accuracy:{}'.format(self.numpy_net.accuracy(test_data)/float(len(test_data))))
                
    
    def vectorized_result(self,j):
        """Return a N-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((self.network_structure[-1], 1))
        e[j] = 1.0
        return e





