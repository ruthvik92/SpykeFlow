import os
import shutil
import sys
from sklearn import preprocessing
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from copy import deepcopy
import scipy.ndimage
import subprocess
from prepdataclass import prepData
sys.path.append(os.getcwd())

prep = prepData(common_path = '../AllDataSets/')
prep.prepMNIST()
prep.prepEMNIST()
prep.prepCIFAR100()
prep.prepCIFAR10()
#######
 # CIFAR10, CIFAR100, MNIST, EMNIST dataset are not in jpeg formats so we need to prep them.
#######
matplotlib.rcParams.update({'font.size': 38})
class InputLayer(object):
    """This class takes images from data sets and converts them into spikes. It returns list of images
        along with their vectorized labels."""

    def __init__(self,val_frac = 0.15,test_frac = 0.15,window_size = 7.0,dataset='caltech101',size = 191,data='train',\
                 debug=False,off_threshold=200,on_threshold=15,border_size=2):

        self.base_path = os.getcwd()
        self.path = os.path.join(self.base_path,'../AllDataSets/'+str(dataset))
        self.train_path = os.path.join(self.base_path,'../AllDataSets/'+str(dataset)+'/train')
        self.test_path = os.path.join(self.base_path,'../AllDataSets/'+str(dataset)+'/test')
        self.val_path = os.path.join(self.base_path,'../AllDataSets/'+str(dataset)+'/valid')

        if not os.path.isdir(self.test_path):
            os.mkdir(self.test_path)

        if not os.path.isdir(self.val_path):
            os.mkdir(self.val_path)
            
        self.val_frac = val_frac
        self.test_frac =test_frac
        self.window_size = window_size
        self.sigma1 = 1.0 
        self.sigma2 = 2.0
        self.dataset = dataset
        try:
            self.categories = os.listdir(self.train_path)  #get the categories to be split
        except:
            print('File path:{} is not found'.format(self.train_path))
            sys.exit()
        #print(self.categories)
        self.size = size
        self.on_dog_filter = self.DoG()
        self.dogNorm = 17
        self.avg_filter = np.ones((self.dogNorm,self.dogNorm))
        self.dim = (self.size, self.size)
        self.tsteps = 10
        self.separation = 2
        self.border_size = border_size
        self.data=data
        self.debug = debug
        self.dog_filter = self.DoG()
        self.off_threshold = off_threshold
        self.on_threshold = on_threshold
    def DataSplitter(self):
        '''
        for datasets like CALTECH101, CALTECH256 or any custom made datasets 
        This code expects the file system like this:

        train
            /category1
            /category2
            /category3
            .
            .
            .
        test
            (empty folder)
        valid
            (empty folder)

        categories will be split according to the specified
        fractions and files will be organuzed like this:
        
        train
            /category1
            /category2
            /category3
            .
            .
            .
        test
            /category1
            /category2
            /category3
        valid
            /category1
            /category2
            /category3

        '''
        for cat in self.categories:
            image_files_path = os.path.join(self.train_path,cat)
            image_files = os.listdir(image_files_path)
            cat_val_path = os.path.join(self.val_path,cat)
            cat_test_path = os.path.join(self.test_path,cat)
            val_test_image_files = image_files[0:int((self.val_frac+self.test_frac)*len(image_files))]
            val_image_files = val_test_image_files[0:int(((self.val_frac)/(self.val_frac+self.test_frac))*len(val_test_image_files))]
            test_image_files = val_test_image_files[int(((self.test_frac)/(self.val_frac+self.test_frac))*len(val_test_image_files)):]
            val_image_files = [os.path.join(image_files_path,items) for items in val_image_files]
            test_image_files = [os.path.join(image_files_path,items) for items in test_image_files]
            if not os.path.isdir(cat_val_path):
                os.mkdir(cat_val_path)
                map(lambda x: shutil.move(x,cat_val_path), val_image_files)

            if not os.path.isdir(cat_test_path):
                os.mkdir(cat_test_path)
                map(lambda x: shutil.move(x,cat_test_path),test_image_files)

            lb = preprocessing.LabelBinarizer()
            
        return lb.fit(self.categories)
    
    def DoG(self):
        x=np.arange(1,int(self.window_size)+1,dtype=np.float64)
        x=np.tile(x,(int(self.window_size),1))
        y=np.transpose(x)
        d2 = ((x-(self.window_size/2))-.5)**2 + ((y-(self.window_size/2))-.5)**2
        gfilter = 1/np.sqrt(2*np.pi) * ( 1/self.sigma1 * np.exp(-d2/2/(self.sigma1**2)) - 1/self.sigma2 * np.exp(-d2/2/(self.sigma2**2)))
        gfilter = gfilter - np.mean(gfilter)
        gfilter = gfilter / np.max(gfilter)
        return gfilter

    def SpikeConversion(self,input_image):

        img = scipy.ndimage.imread(input_image,mode='L')
        #img=mpimg.imread(input_image)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #throws an error when a gray scale is encountered(BUG)
        img = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
        img = img.astype(np.float32)
        
        on_dog_result = cv2.filter2D(img,-1,self.dog_filter)
        off_dog_result=-on_dog_result
        temp_on = deepcopy(on_dog_result)
        temp_off = deepcopy(off_dog_result)

        border = np.zeros(on_dog_result.shape)
        border[self.border_size:-self.border_size,self.border_size:-self.border_size]=1
        on_dog_result = on_dog_result*border
        off_dog_result = off_dog_result*border
        
        on_dog_result[on_dog_result<self.on_threshold]=0
        off_dog_result[off_dog_result<self.off_threshold]=0

        

        on_dog_result = on_dog_result.round(decimals=5)
        off_dog_result = off_dog_result.round(decimals=5)
        
    
        
        avg_on_dog_result = (cv2.filter2D(on_dog_result,-1,self.avg_filter)+0.0001)/self.dogNorm**2
        avg_off_dog_result = (cv2.filter2D(off_dog_result,-1,self.avg_filter)+0.0001)/self.dogNorm**2
        on_dog_result/=avg_on_dog_result
        off_dog_result/=avg_off_dog_result

        on_dog_result = 1./(on_dog_result.flatten())
        off_dog_result = 1./(off_dog_result.flatten())

        on_indices=np.argsort(on_dog_result)
        on_latencies = on_dog_result[on_indices]
        on_inf_indices = np.where(on_latencies==np.inf)
        on_latencies = np.delete(on_latencies,on_inf_indices)
        on_indices =np.delete(on_indices,on_inf_indices)
        
        off_indices=np.argsort(off_dog_result)
        off_latencies = off_dog_result[off_indices]
        off_inf_indices = np.where(off_latencies==np.inf)
        off_latencies = np.delete(off_latencies,off_inf_indices)
        off_indices =np.delete(off_indices,off_inf_indices)

        row = on_indices/self.size
        col = on_indices%self.size
        t_axis = (np.ceil(np.arange(1,on_indices.size+1)/(float(on_indices.size)/(self.tsteps)))-1).astype(np.uint)
        sparse_array = np.zeros((self.size,self.size,2,(self.tsteps+self.separation)),dtype =np.bool_)
        sparse_array[row,col,0,t_axis]=1
        

        row = off_indices/self.size
        col = off_indices%self.size
        t_axis = (np.ceil(np.arange(1,off_indices.size+1)/(float(off_indices.size)/(self.tsteps)))-1).astype(np.uint)
        sparse_array[row,col,1,t_axis]=1

        if(self.debug):
            print('Number of ON spikes:{}'.format(sparse_array[:,:,0,:].sum()))
            print('Number of OFF spikes:{}'.format(sparse_array[:,:,1,:].sum()))
            fig, axes = plt.subplots(2,3)
            axes = axes.flatten()
            
            axes[0].imshow(img,cmap='gray')
            axes[0].tick_params(labelsize=26)
            axes[0].set_title('Original Image',fontsize=26)

            axes[1].imshow(temp_on,cmap='gray')
            axes[1].tick_params(labelsize=26)
            axes[1].set_title('ON DoG',fontsize=26)

            axes[2].imshow(sparse_array[:,:,0,:].sum(axis=2),cmap='gray')
            axes[2].tick_params(labelsize=26)
            axes[2].set_title('Reconstruction',fontsize=26)

            axes[3].imshow(img,cmap='gray')
            axes[3].tick_params(labelsize=26)
            axes[3].set_title('Original Image',fontsize=26)
            
            axes[4].imshow(temp_off,cmap='gray')
            axes[4].tick_params(labelsize=26)
            axes[4].set_title('OFF DoG',fontsize=26)

            axes[5].imshow(sparse_array[:,:,1,:].sum(axis=2),cmap='gray')
            axes[5].tick_params(labelsize=26)
            axes[5].set_title('Reconstruction',fontsize=26)
            plt.show()
            
        return sparse_array
        
    def EncodedData(self):
        label_creator = self.DataSplitter()
        path = os.path.join(self.path,self.data)
        final_data = []
        
        if(self.debug):
            fig, axes = plt.subplots(1,2)
            axes = axes.flat
            im1=axes[0].imshow(self.dog_filter,interpolation='none')
            axes[0].set_title('ON Center filter')
            im2=axes[1].imshow(-self.dog_filter,interpolation='none')
            axes[1].set_title('OFF Center filter')
            fig.subplots_adjust(right=0.8)
            fig.colorbar(im1, ax=axes[0],fraction=0.046,pad=0.04)
            fig.colorbar(im2, ax=axes[1],fraction=0.046,pad=0.04)
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            plt.show()
            
        for category in self.categories:
            category_path = os.path.join(path,category)
            category_files = os.listdir(category_path)
            category_files = [os.path.join(category_path,items) for items in category_files]
            category_labels = label_creator.transform([category])
            #print(category_labels)
            #sys.exit()
            ncoded_category = map(lambda x: self.SpikeConversion(x),category_files)
            ncoded_category = [(items,category_labels) for items in ncoded_category]
            final_data.extend(ncoded_category)

        return final_data

    







