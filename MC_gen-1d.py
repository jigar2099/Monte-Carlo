import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks
sns.set(style='darkgrid')

class monte_carlo_simulation:
    def __init__(self, path, threshold, stepsize, peak_per_sample):
        self.path = path
        self.threshold = threshold
        self.stepsize = stepsize
        self.peak_per_sample = peak_per_sample
    def sep_mat(self):
        arr = np.loadtxt('{}'.format(self.path))
        arr = np.where(arr<0, 0, arr)
        arr_sub = arr - self.threshold
        arr_w = np.where(arr_sub<0,0,arr_sub)
        arr_threshold = arr_w.reshape(np.int(arr_w.shape[0]/self.stepsize), self.stepsize)
        
        rate_l = []
        ind_l = []
        val_l = []
        for i in range(arr_threshold.shape[0]):
            peak_ind, peak_val = find_peaks(arr_threshold[i], height=0)
            rate_l.append(len(peak_ind))
            ind_l.append(peak_ind)
            val_l.append(peak_val)
        rate_init = np.asarray(rate_l)
        ind_init = np.asarray(ind_l)
        val_init = np.asarray(val_l)
        
        '''Now delete all those samples which have sliced signals'''
        first_non_zero_ind = np.argwhere(arr_threshold[:,0]!=0) # get indices of samples with slice at the begining
        mod_arr1 = np.delete(arr_threshold,first_non_zero_ind,0) # delete those samples
        last_non_zero_ind = np.argwhere(mod_arr1[:,99]!=0) # from remaining samples get those with slice at at the end
        mod_arr2 = np.delete(mod_arr1,last_non_zero_ind,0) # remove those samples
        
        '''Now make seperate those samples which have our selected rate, and make their individual matrix'''
        rate_side = []
        ind_side = []
        val_side = []
        for i in range(mod_arr2.shape[0]):
            peak_ind, peak_val = find_peaks(mod_arr2[i],height=0)
            rate_side.append(len(peak_ind))
            ind_side.append(peak_ind)
            val_side.append(peak_val)
        rate_side = np.asarray(rate_side)
        ind_side = np.asarray(ind_side)
        val_side = np.asarray(val_side)
        
        rate_side_ind = np.argwhere(rate_side==self.peak_per_sample) # indices where there is required #photons in rate_side samples
        x = mod_arr2[rate_side_ind,:] # using row index create matrix whose samples have required #photons
        x = x.reshape(x.shape[0],self.stepsize)
        return x, arr_threshold, rate_init
    
    def complete_mc(self,required_rate,multiple):
        arr = np.loadtxt('{}'.format(self.path))
        arr = np.where(arr<0, 0, arr)
        arr_sub = arr - self.threshold
        arr_w = np.where(arr_sub<0,0,arr_sub)
        arr_threshold = arr_w.reshape(np.int(arr_w.shape[0]/self.stepsize), self.stepsize)
        
        rate_l = []
        ind_l = []
        val_l = []
        for i in range(arr_threshold.shape[0]):
            peak_ind, peak_val = find_peaks(arr_threshold[i], height=0)
            rate_l.append(len(peak_ind))
            ind_l.append(peak_ind)
            val_l.append(peak_val)
        rate_init = np.asarray(rate_l)
        ind_init = np.asarray(ind_l)
        val_init = np.asarray(val_l)
        
        '''Now delete all those samples which have sliced signals'''
        first_non_zero_ind = np.argwhere(arr_threshold[:,0]!=0) # get indices of samples with slice at the begining
        mod_arr1 = np.delete(arr_threshold,first_non_zero_ind,0) # delete those samples
        last_non_zero_ind = np.argwhere(mod_arr1[:,99]!=0) # from remaining samples get those with slice at at the end
        mod_arr2 = np.delete(mod_arr1,last_non_zero_ind,0) # remove those samples
        
        '''Now make seperate those samples which have our selected rate, and make their individual matrix'''
        rate_side = []
        ind_side = []
        val_side = []
        for i in range(mod_arr2.shape[0]):
            peak_ind, peak_val = find_peaks(mod_arr2[i],height=0)
            rate_side.append(len(peak_ind))
            ind_side.append(peak_ind)
            val_side.append(peak_val)
        rate_side = np.asarray(rate_side)
        ind_side = np.asarray(ind_side)
        val_side = np.asarray(val_side)
        
        rate_side_ind = np.argwhere(rate_side==self.peak_per_sample) # indices where there is required #photons in rate_side samples
        x = mod_arr2[rate_side_ind,:] # using row index create matrix whose samples have required #photons
        x = x.reshape(x.shape[0],self.stepsize)
        
        required_rate = required_rate
        original_samples = np.random.randint(0,arr_threshold.shape[0],multiple*arr_threshold.shape[0])
        rate = []
        for i in original_samples:
            peak_ind, peak_pos = find_peaks(arr_threshold[i],height=0)
            rate = len(peak_ind)
            sub = required_rate-rate
            ind = peak_pos.items()
            data_pos = list(ind)
            peak_position = np.array(data_pos)
            if sub>0:
                sep_matrix = np.random.randint(0,x.shape[0],sub)# get random numbers for row-inds(matrix with samples of 1 photon) a.c.w sub
                for k in sep_matrix:
                    time_step = np.random.randint(0,100-len(x[k][x[k]!=0]),1)# cut the shape from randomly selected sample with only 1 photon, and get step-ind for original sample, where to place it
                    for l in time_step:
                        non_zero = len(x[k][x[k]!=0])# select only non-zero part of a photon shape
                        arr_threshold[i][l:l+non_zero]+=x[k][x[k]!=0] # modify original sample with selected sample
                        rate_init[i]+=1 # modify original rate
            else:
                pass
            #a = np.concatenate((peak_position[0][1], time_step),axis=0).astype(int)
        return arr_threshold, rate_init, x
        
    
    def mc_modification(self,required_rate,multiple, arr_Threshold, rate_Init, X):
        required_rate = required_rate
        original_samples = np.random.randint(0,arr_Threshold.shape[0],multiple*arr_Threshold.shape[0])
        rate = []
        for i in original_samples:
            peak_ind, peak_pos = find_peaks(arr_Threshold[i],height=0)
            rate = len(peak_ind)
            sub = required_rate-rate
            ind = peak_pos.items()
            data_pos = list(ind)
            peak_position = np.array(data_pos)
            if sub>0:
                sep_matrix = np.random.randint(0,X.shape[0],sub)# get random numbers for row-inds(matrix with samples of 1 photon) a.c.w sub
                for k in sep_matrix:
                    time_step = np.random.randint(0,100-len(X[k][X[k]!=0]),1)# cut the shape from randomly selected sample with only 1 photon, and get step-ind for original sample, where to place it
                    for l in time_step:
                        non_zero = len(X[k][X[k]!=0])# select only non-zero part of a photon shape
                        arr_Threshold[i][l:l+non_zero]+=X[k][X[k]!=0] # modify original sample with selected sample
                        rate_Init[i]+=1 # modify original rate
            else:
                pass
            #a = np.concatenate((peak_position[0][1], time_step),axis=0).astype(int)
        return arr_Threshold, rate_Init
        

#if __name__ == "__main__":
#    di_path = '/home/ii_pro/bigdisk/jkb2/Monte_carlo_26oct/merged_slg_creation/mLED_3-11.txt'
#    f1 = monte_carlo_simulation(di_path,1,100,1)
#    f2,f3,f4 = f1.sep_mat()
    
#    f10 = monte_carlo_simulation(di_path,1,100,1)
#    f20,f30,f40 = f10.sep_mat()
#    f5,f6 = f10.mc_modification(5,1,f30,f40,f20)
    #f3 = filter_matrix.sep_mat(di_path,1,100,1)
