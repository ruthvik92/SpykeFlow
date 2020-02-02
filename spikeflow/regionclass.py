'''Given a grid of N**2 numbers arranged in a map NxN, a kernel of MxM such that M<N and a point, p this code generates a region of MxM centered at point p.
    -> Firstly, row and column of the supplied point are calculated.
    -> Then, based on the location of the point, number of rows above and below the point that lie in the region are calculated.
    -> Above step is repeated for columns also, columns are left and right.
    -> Points in the first row of region are calculated using upper row number and left column number and upper row number and right column
    -> Then, based on difference between upper row and down row numbers, additional rows are appended to the region
    This code was tested for grid_size = 10,9 kernel_size= 5,5 respectively
'''
import numpy as np
class genregion(object):

    def __init__(self, point, grid_size, kernel_size):
        self.point = point
        self.grid_size = grid_size
        self.kernel_size = kernel_size
        self.r = 0
        self.c = 0
        self.tar_ur = 0
        self.tar_dr = 0
        self.tar_lc = 0
        self.tar_rc = 0
        self.a_row = []
    def GenRegion(self):
        
        self.r = self.point / self.grid_size
        self.c = self.point % self.grid_size

        self.tar_ur = self.r - self.kernel_size/2
        if(self.tar_ur <0):
            self.tar_ur = 0

        self.tar_dr = self.r + self.kernel_size/2
        if(self.tar_dr >= self.grid_size):
            self.tar_dr = self.grid_size-1

        self.tar_lc = self.c - self.kernel_size/2
        if(self.tar_lc < 0):
            self.tar_lc = 0

        self.tar_rc = self.c + self.kernel_size/2
        if(self.tar_rc >= self.grid_size):
            self.tar_rc = self.grid_size-1
            
        self.a_row = [i for i in range(self.tar_ur*(self.grid_size)+self.tar_lc, self.tar_ur*(self.grid_size)+self.tar_rc+1)]
        temp = self.a_row
        self.a_row= [temp]
        for i in range(1,self.tar_dr-self.tar_ur+1):
            self.a_row.append([items+self.grid_size*i for items in temp])

        return np.asarray(self.a_row)
