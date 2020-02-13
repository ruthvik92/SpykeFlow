
import math
import math
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

image_size = 27  
jaer_size =0
class poolconv():
    
    '''
        :param window: Size of the window
        :param mapp:   Size of the map
        :param weight_a: lower bound of random uniform distb for weights(or mean of a normal RV)
        :param weight: upper bound of random uniform distb for weights(or SD of an RV)
        :delay: Axonal delay
        :overlap: Overlap between windows

           code tested for   |map     |window    |
                             |7       | 3 ov 1   |
                             |9       | 3 ov 1   |                         
                             |10      | 3 ov1 NW |
                             |10      | 3 ov 0   |
                        
            eg: here maps=5 means it's a 5x5 and window=2 means 2x2

                              - - - - - -
                              | | | | | | -
                              - - - - - -   -
                              | | | | | |     -
                              - - - - - -       -        - - - 
                              | | | | | |          }---> | | |
                              - - - - - -       -        - - -
                              | | | | | |     -          | | |
                              - - - - - -   -            - - -
                              | | | | | | - 
                              - - - - - -
  - -
  | | each of these boxes is a neuron, in this example we took window of 2x2 of neurons on leftside squares to form one 
  square on rightside squares. There is no overlap and one column on right and one row on bottom of leftside squares
  - -
    arguments are window, mapp, weight,delay,overlap
    The structure of list here is [pre_syn_neuron, post_syn_neuron, weight, axonal_delay]
    [(0, 0, 1, 1), (1, 0, 1, 1), (4, 0, 1, 1), (5, 0, 1, 1), (2, 1, 1, 1), (3, 1, 1, 1), (6, 1, 1, 1), (7, 1, 1, 1),
    (8, 2, 1, 1), (9, 2, 1, 1), (12, 2, 1, 1), (13, 2, 1, 1), (10, 3, 1, 1), (11, 3, 1, 1), (14, 3, 1, 1), (15, 3, 1, 1)]
    This code can also be used for convolving populations. For performing convolution just enter the overlap value:
    window-1. ORIGINNALY THIS CLASS WAS WRITTEN TO GENERATE THE CONVOLUTION CONNECTOR FOR PYNN+NEURON OR PYNN+BRIAN. I DONT USE
    ANY OF THOSE SO, I SLIGHTLY MODIFIED THIS TO BE ABLE TO RETURN A MAPPING OF NEURONS FROM PRE-SYNAPTIC TO POST SYNAPTIC.
    METHODS aerfile_to_spike, raster_plot_spike ARE TAKEN FROM GARIBALDI PINEDA'S GITHUB ACCOUNT (UNI OF MANCHESTER, SPINNAKER GROUP) 
    
    '''
    
    #conn_list = [(0,0,1,1,1)]
    def __init__(self,window,mapp,weight_a,weight,delay,overlap):
        self.window = window
        self.mapp = mapp
        self.weight_a = weight_a
        self.weight = weight
        self.delay = delay
        self.overlap = overlap	
        self.conn_list = [(0,0,random.normalvariate(self.weight_a,self.weight),1)] 
        self.only_weights = []
        self.only_indices=[]
    
    
    def PoolConv(self):

        if(self.window>self.mapp):
            print('window:{} is greater than mapp:{}').format(self.window,self.mapp)
            sys.exit()
        ''' this method implements the logic for pooling/convolution. If overlap is window-1 then it's convolution.'''
        extra = int(math.floor((self.mapp-self.window)/(self.window-self.overlap)+1)) 
        #random.seed()
        
        for p in range(0,self.window-1):   
            w = self.conn_list[p][0]+1  
            v = self.conn_list[p][1]
            q = random.normalvariate(self.weight_a,self.weight)
            if(q>1.0 or q <0): #Because 1 is only 4 S.Ds away from 0.8 there is a possibility that we might get w>1
                #print 'SCREEEEEEEEEEEEEEEEECH!!!!!!q>1, so I set it to 1.0'
                #print q
                q=0.8
                #sys.exit()
            s = self.delay
            self.conn_list.append((w,v,q,s))
            #print "executing"

        for c in range(0,(self.window-1)*self.window):    
            x = self.conn_list[c][0] + self.mapp  
            y = self.conn_list[c][1]
            q = random.normalvariate(self.weight_a,self.weight)
            if(q>1.0 or q<0 ): 
                #print 'SCREEEEEEEEEEEEEEEEECH!!!!! q>1, so I set it to 0.8'
                #print q
                q=0.8
                #sys.exit()
            s = self.delay
            self.conn_list.append((x,y,q,s))

        skip = []
        for p in range(0,extra*extra):
            skip.append((p+1)*extra-1)    
        
                                                           
        for i in range(0,extra*extra-1):     
            if(i in skip):                 
                z = skip.index(i)
                for k in range(0+(self.window**2)*i,(self.window**2)+(self.window**2)*i):   
                    z = self.conn_list[(self.window**2)*skip[0]][0]
                    n = (self.conn_list[k][0] + self.mapp*(self.window-self.overlap)) -z   
                    m = self.conn_list[k][1] + 1  
                    q = random.normalvariate(self.weight_a,self.weight)
                    
                    if(q>1.0 or q<0): 
                        #print 'SCREEEEEEEEEEEEEEEEECH!!!!!!!!q>1, so I set it to 0.8'
                        #print q
                        q=0.8
                        #sys.exit()
                    s = self.delay
                    new_item= (n,m,q,s)
                    self.conn_list.append(new_item)             

            else:
                for j in range(0+(self.window**2)*i,(self.window**2)+(self.window**2)*i):
                    a = self.conn_list[j][0] +self.window-self.overlap     
                    b = self.conn_list[j][1] +1
                    q = random.normalvariate(self.weight_a,self.weight)
                    if(q>1.0 or q<0): 
                        #print 'SCREEEEEEEEEEEEEEEEECH!!!!!!!q>1, so I set it to 0.8'
                        #print q
                        q=0.8
                        #sys.exit()
                    s = self.delay
                    list1 = (a,b,q,s)
                    self.conn_list.append(list1)
        #return self.conn_list

    def IndicesWeights(self):
        self.PoolConv()
        indices = []
        weights = []
        for i in range((self.mapp-self.window+1)**2):
            indices.append(np.asarray([items[0] for items in self.conn_list[i*(self.window**2):(i+1)*(self.window**2)]]))
            weights.append(np.asarray([items[2] for items in self.conn_list[i*(self.window**2):(i+1)*(self.window**2)]]))
         
        return indices, weights
        


