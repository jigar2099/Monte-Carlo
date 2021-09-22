import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class noisy_gaussian_image:
    
    def __init__(self, pixel, mu1, mu2, cov_mat00, cov_mat11, src, bg):
        self.pixel = pixel
        self.mu1 = mu1
        self.mu2 = mu2
        self.cov_mat00 = cov_mat00
        self.cov_mat11 = cov_mat11
        self.src = src
        self.bg = bg
        
        
    def base(self):
    # introduce grid-base for image
        x = np.linspace(0,self.pixel-1,self.pixel)
        y = np.linspace(0,self.pixel-1,self.pixel)
        x, y = np.meshgrid(x, y)#-------------->establishment of grids for image
        pos = np.dstack((x,y))#----------------> dstacking provides stacking of x and y grids on top of each other

        # generate matrix of mean and covariance_matrix
        mu = np.array([self.mu1,self.mu2])
        cov_mat = np.array([[self.cov_mat00, 0],[0, self.cov_mat11]])
        bg_grid = np.dstack((x,y))
    
        return pos, mu, cov_mat, bg_grid

    def noisy_gaussian_multivariate(self, pos, mu, cov_mat,src,bg):
        n = mu.shape[0]
        det_sig = np.linalg.det(cov_mat)
        inv_sig = np.linalg.inv(cov_mat)

        powerr = np.einsum('ijk,kl,ijl->ij', pos-mu, inv_sig, pos-mu)
        #powerr = np.einsum('...k,kl,...l->...', pos-mu, inv_sig, pos-mu)fig.update_layout(title_text="Ring cyclide")
        denominator = np.sqrt(np.power((2*np.pi),n) * det_sig)
        out = self.src*np.exp(-powerr/2)/denominator
        #print(out.shape)
        bg_grid = np.random.poisson(bg,size=out.shape)
        #out = out.astype(np.float64)
        #return out
        #A_b = A_s*bg
        return np.random.poisson(out), bg_grid
    