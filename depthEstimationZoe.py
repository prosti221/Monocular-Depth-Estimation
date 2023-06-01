import torch
import cv2
import numpy as np
from PIL import Image
'''
Uses a pre-trained ZoeDepth model to estimate the depth of a monocular image.
In contrast to MiDaS, this model attempts to recover the metric scale, but often it is way off the true scale.
It does however produce depth maps with more detail.
Don't even bother running this on CPU, its even slow-ish on GPU. 
'''
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DepthEstimatorZoe:
    def __init__(self, model_type='NK', device='cpu'):
        self.device = device # cpu or cuda
        self.model_type = model_type

        torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)
        repo = "isl-org/ZoeDepth"

        if self.model_type == 'NK':
            self.model = torch.hub.load(repo, "ZoeD_NK", pretrained=True, force_reload=False, config_mode='eval').to(self.device)
        elif self.model_type == 'K':
            self.model = torch.hub.load(repo, "ZoeD_K", pretrained=True, force_reload=False, config_mode='eval').to(self.device)
        elif self.model_type == 'N':
            self.model = torch.hub.load(repo, "ZoeD_N", pretrained=True, force_reload=False, config_mode='eval').to(self.device)
        self.model.eval()
        self.scale_factors = []
        self.outlier_count = 0
        self.N = 15 # number of points to use for averaging
        self.reset_thresh = 35 

    def predict_depth(self, img):
        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth_numpy = self.model.infer_pil(inp)
        return depth_numpy

    def inv_depth_to_depth(self, inv_map):
        return 1 / (inv_map + 1e-6) # add small epsilon to avoid division by zero

    def updateDepthEstimates(self, depthMap, knownMeasurements):                
        '''                                                                     
        Depth map estimate update                                               
                                                                                
        Updates a depth map based on known measurements across the depth range. 
                                                                                
            Parameters:                                                         
                depthMap (MxNx1 numpy array) : numpy matrix of estimated depth map.
                knownMeasurements: Nx3 numpy matrix, N points with [x, y, z] values.
                degree (int) : number of degrees for the polynomial regression model
                                                                                
            Returns:                                                            
                updatedDepthMap (MxNx1 numpy array) : numpy matrix of estimated depth map.
        '''                                                                     
        if knownMeasurements is None:                                           
            return depthMap * self.scale_factors[-1] if len(self.scale_factors) != 0 else depthMap
        knownMeasurements_idx = knownMeasurements[:, 0:2].astype(int)           
        knownMeasurements = knownMeasurements[:, 2]                             
                                                                                
        if(len(knownMeasurements) < 1):                                         
            print("No correction points")                                          
            return depthMap * self.scale_factors[-1] if len(self.scale_factors) != 0 else depthMap
                                                                                   
        elif(len(knownMeasurements) == 1):                                         
            depthValue = depthMap[knownMeasurements_idx[0, 0], knownMeasurements_idx[0, 1]]
            scalingFactor = knownMeasurements[0]/depthValue                     
            print("scaling factor =", knownMeasurements[0], "/", depthValue, "=", scalingFactor)
            updatedDepthMap = depthMap * scalingFactor                          
                                                                                
        else:                                                                   
            #get known measurements (normally retrived through object detection)
            true_depths = knownMeasurements.reshape(-1, 1)                      
                                                                                
            #find the corresponding points in the depth map estimate            
            corresponding_depths = depthMap[knownMeasurements_idx[:, 0], knownMeasurements_idx[:, 1]].reshape(-1, 1)
                                                                                
            x, _, _, _ = np.linalg.lstsq(corresponding_depths, true_depths, rcond=None)
                                                                                
            # check if scale factor is an outlier                               
            if not self.is_scale_factor_outlier(x[0]):                          
                self.outlier_count = 0                                          
                print("scaling factor =", x[0])                                 
                self.scale_factors.append(x[0])                                 
                print("Mean scaling factor =", np.mean(self.scale_factors))     
                #updatedDepthMap = depthMap * x[0]                              
            else:                                                               
                if self.outlier_count < self.reset_thresh:                      
                    self.outlier_count += 1                                     
                else:                                                           
                    self.outlier_count = 0                                      
                    last_scale_factor_mean = np.mean(self.scale_factors[-self.N:]) if len(self.scale_factors) > self.N else np.mean(self.scale_factors)
                    self.scale_factors = []                                     
                    return depthMap * last_scale_factor_mean                    
                print(f"outlier detected: {x[0]}")                              

            updatedDepthMap = depthMap * np.mean(self.scale_factors[-self.N:]) if len(self.scale_factors) > self.N else depthMap * np.mean(self.scale_factors)
                                                                                
                                                                                
        return updatedDepthMap   

    def is_scale_factor_outlier(self, current_scale_factor, num_stddevs=2.0):
        """
        Check if the current scale factor is an outlier relative to the previous scale factors.
        :param scale_factors: List of previous scale factors.
        :param current_scale_factor: Scale factor for the current time step.
        :param num_stddevs: Number of standard deviations to consider as an outlier.
        :return: True if the current scale factor is an outlier, False otherwise.
        """
        if len(self.scale_factors) < 10:
            return False  # Not enough data to determine if current scale factor is an outlier.

        # Calculate the mean and standard deviation of the previous scale factors.
        mean = np.mean(self.scale_factors)
        std = np.std(self.scale_factors)

        # Check if the current scale factor is more than num_stddevs standard deviations away from the mean.
        if abs(current_scale_factor - mean) > num_stddevs * std:
            return True  # Current scale factor is an outlier.
        else:
            return False  # Current scale factor is not an outlier.
