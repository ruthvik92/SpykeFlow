import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import theano
import cPickle
import gzip


class prepData(object):
    """Data sets like EMNIST, MNIST, CIFAR10 and CIFAR100 come in pickled zip files
        this class will extract them to folders based on the class labels just so the
        input data to the code is in same fashion as in CALTECH101, CALTECH256, NMNIST,
        NCALTECH101"""

    def __init__(self,common_path):
        self.common_path = common_path

    def create_images(self,data,nofClasses,dataset):
        common_path = self.common_path+dataset+'train'

        if not os.path.isdir(common_path):
            os.mkdir(common_path)

        if(len(os.listdir(common_path))==0):
            map(lambda x: os.mkdir(os.path.join(common_path,str(x))), range(nofClasses))
            x,y = data
            y=np.array(y).astype(np.int)
            x=np.array(x)
            for clas in range(nofClasses):
                pos = np.where(y==clas)
                digits = x[pos[0]]
                map(lambda x,y: plt.imsave(os.path.join(common_path,str(clas),str(x)),y),range(len(digits)),digits)
    
        return
############################  MNIST DATA ####################################
    def get_train_data(self):
        '''Extracts images and labels from the train files obtained from
           http://yann.lecun.com/exdb/mnist/ (This is from gary's poisson_tools.py code)
           
           :returns: A tuple containing arrays of the images (train_x) and
                     labels (train_y).
        '''
        common_path = '../AllDataSets/mnist/'
        file_name = common_path+'train-images.idx3-ubyte'
        f = open(file_name, "rb")
        magic_number, list_size, image_hight, image_width  = np.fromfile(f, dtype='>i4', count=4)
        train_x = np.fromfile(f, dtype='>u1', count=list_size*image_hight*image_width)
        train_x = np.reshape(train_x, (list_size,image_hight*image_width))
        f.close()
        
        file_name = common_path+'train-labels.idx1-ubyte'
        f = open(file_name, "rb")
        magic_number, list_size = np.fromfile(f, dtype='>i4', count=2)
        train_y = np.fromfile(f, dtype='>u1', count=list_size*image_hight*image_width)
        f.close()
        
        return np.double(train_x), np.double(train_y)


    def get_test_data(self):
        '''Extracts images and labels from the test files obtained from
           http://yann.lecun.com/exdb/mnist/
           
           :returns: A tuple containing arrays of the images (test_x) and
                     labels (test_y).
        '''
        common_path = '../AllDataSets/mnist/'
        file_name = common_path+'t10k-images.idx3-ubyte'
        f = open(file_name, "rb")
        magic_number, list_size, image_hight, image_width  = np.fromfile(f, dtype='>i4', count=4)
        test_x = np.fromfile(f, dtype='>u1', count=list_size*image_hight*image_width)
        test_x = np.reshape(test_x, (list_size,image_hight*image_width))
        f.close()
        
        file_name = common_path+'t10k-labels.idx1-ubyte'
        f = open(file_name, "rb")
        magic_number, list_size = np.fromfile(f, dtype='>i4', count=2)
        test_y = np.fromfile(f, dtype='>u1', count=list_size*image_hight*image_width)
        f.close()
        
        return np.double(test_x), np.double(test_y)

    def prepMNIST(self):
        print('Prepping the MNIST data .... \n')
        try:
            train_x, train_y = self.get_train_data()
            train_x_list = map(lambda x: np.reshape(x,(28,28)),train_x)
            train_y_list = train_y.tolist()

            test_x, test_y = self.get_test_data()
            test_x_list = map(lambda x: np.reshape(x,(28,28)),test_x)
            test_y_list = test_y.tolist()

            train_x_list.extend(test_x_list)
            train_y_list.extend(test_y_list)
            nofClasses = 10

            self.create_images((train_x_list,train_y_list),nofClasses,dataset='mnist/')
            print('Finished MNIST data ..\n')
        except:
            print('Unable to find data in AllDataSets/mnist..')
########################  EMNIST DATA ######################################


    def load_data(self):
        """Return the MNIST data as a tuple containing the training data,
        the validation data, and the test data.

        The ``training_data`` is returned as a tuple with two entries.
        The first entry contains the actual training images.  This is a
        numpy ndarray with 50,000 entries.  Each entry is, in turn, a
        numpy ndarray with 784 values, representing the 28 * 28 = 784
        pixels in a single MNIST image.

        The second entry in the ``training_data`` tuple is a numpy ndarray
        containing 50,000 entries.  Those entries are just the digit
        values (0...9) for the corresponding images contained in the first
        entry of the tuple.

        The ``validation_data`` and ``test_data`` are similar, except
        each contains only 10,000 images.

        This is a nice data format, but for use in neural networks it's
        helpful to modify the format of the ``training_data`` a little.
        That's done in the wrapper function ``load_data_wrapper()``, see
        below.
        """
        f = gzip.open('../AllDataSets/emnist/emnist-balanced.pkl.gz', 'rb')
        training_data, validation_data, test_data = cPickle.load(f)
        f.close()

        return (training_data, validation_data, test_data)

    def prepEMNIST(self):
        print('Prepping the EMNIST data .... \n')
        try:
            training_data, validation_data, test_data = self.load_data()
            #print(len(training_data[0]), len(validation_data[0]), len(test_data[0]))
            #sys.exit()
            train_x, train_y = training_data
            train_x_list = map(lambda x: np.reshape(x,(28,28)),train_x)
            train_y_list = train_y.tolist()

            val_x, val_y = validation_data
            val_x_list = map(lambda x: np.reshape(x,(28,28)),val_x)
            val_y_list = val_y.tolist()
            train_x_list.extend(val_x_list)
            train_y_list.extend(val_y_list)


            test_x, test_y = test_data
            test_x_list = map(lambda x: np.reshape(x,(28,28)),test_x)
            test_y_list = test_y.tolist()
            train_x_list.extend(test_x_list)
            train_y_list.extend(test_y_list)

            nofClasses = 47
            self.create_images((train_x_list,train_y_list),nofClasses,dataset='emnist/')
            print('Finished EMNIST data ..\n')
        except:
            print('Unable to find data in AllDataSets/emnist..')
            
############################### CIFAR 10 DATA #########################################

    def unpickle(self,file):
        import cPickle
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict

    def prepCIFAR10(self):
        print('Prepping the CIFAR10 data .... \n')
        try:
            path = '../AllDataSets/cifar10/'
            data_pkl = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
            data = []
            labels = []
            batch_label = []
            file_names = []
            for file in data_pkl:
                open(path+file,'rb')
                batch = self.unpickle(path+file)
                data.append(batch['data'])
                labels.append(batch['labels'])
                batch_label.append(batch['batch_label'])
                file_names.append(batch['filenames'])


            all_data = np.vstack(data).reshape(60000,3,32,32) #reshaping it to 60000,32,32,3 will messup the image
            all_data = theano.shared(all_data)
            all_data = all_data.dimshuffle(0,2,3,1)
            train_x_list = all_data.eval()
            train_y_list = np.hstack(labels)

            nofClasses = 10
            self.create_images((train_x_list,train_y_list),nofClasses,dataset='cifar10/')
            print('Finished CIFAR10 data ..\n')
        except:
            print('Unable to find data in AllDataSets/cifar10..')
############################### CIFAR 100 DATA #########################################

    def prepCIFAR100(self):
        try:
            print('Prepping the CIFAR100 data .... \n')
            path = '../AllDataSets/cifar100/'
            data_pkl = ['train_data','test_data']
            data = []
            labels = []
            batch_label = []
            file_names = []
            for file in data_pkl:
                open(path+file,'rb')
                batch = self.unpickle(path+file)
                data.append(batch['data'])
                labels.append(batch['fine_labels'])
                batch_label.append(batch['batch_label'])
                file_names.append(batch['filenames'])



            all_data = np.vstack(data).reshape(60000,3,32,32) #reshaping it to 60000,32,32,3 will messup the image
            all_data = theano.shared(all_data)
            all_data = all_data.dimshuffle(0,2,3,1)
            train_x_list = all_data.eval()
            train_y_list = np.hstack(labels)

            nofClasses = 100
            self.create_images((train_x_list,train_y_list),nofClasses,dataset='cifar100/')
            print('Finished CIFAR100 data ..\n')
        except:
            print('Unable to find data in AllDataSets/cifar100..') 
prep = prepData(common_path = '../AllDataSets/')
