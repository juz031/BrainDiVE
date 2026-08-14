import struct
import numpy as np
import math
import copy
import scipy.stats
import pandas as pd
import os


def make_log_polar_grid(sigma_range=[0.02, 1], n_sigma_steps=10, \
                          eccen_range=[0, 7/8.4], n_eccen_steps=10, n_angle_steps=16):
    
    """
    Create a grid of pRF positions/sizes that are evenly spaced in polar angle.
    Sizes and eccentricities are logarithmically spaced.
    Units for size/sigma are relative to image aperture size (1.0 units)
    To convert to degrees, multiply these values by degrees of display (8.4 degrees for NSD expts)
    """

    sigma_vals = np.logspace(np.log(sigma_range[0]), np.log(sigma_range[1]), \
                             base=np.e, num=n_sigma_steps)
    # min eccen should usually be zero, accounting for this here
    small_value = 0.10
    eccen_vals = np.logspace(np.log(eccen_range[0]+small_value), np.log(eccen_range[1]+small_value), \
                             base=np.e, num=n_eccen_steps) - small_value
    angle_step = 2*np.pi/n_angle_steps
    angle_vals = np.linspace(0,np.pi*2-angle_step, n_angle_steps)

    eccen_vals = eccen_vals.astype(np.float32)
    angle_vals = angle_vals.astype(np.float32)
    sigma_vals = sigma_vals.astype(np.float32)

    # First, create the grid of all possible combinations.
    x_vals = (eccen_vals[:,np.newaxis] * np.cos(angle_vals[np.newaxis,:]))
    y_vals = (eccen_vals[:,np.newaxis] * np.sin(angle_vals[np.newaxis,:]))

    x_vals = np.tile(np.reshape(x_vals, [n_angle_steps*n_eccen_steps, 1]), [n_sigma_steps,1])
    y_vals = np.tile(np.reshape(y_vals, [n_angle_steps*n_eccen_steps, 1]), [n_sigma_steps,1])

    sigma_vals = np.repeat(sigma_vals, n_angle_steps*n_eccen_steps)[:,np.newaxis]

    prf_params = np.concatenate([x_vals, y_vals, sigma_vals], axis=1)

    # Now removing a few of these that we don't want to actually use - some duplicates, 
    # and some that go entirely outside the image region.
    unrows, inds = np.unique(prf_params, axis=0, return_index=True)
    prf_params = prf_params[np.sort(inds),:]

    # what is the approx spatial extent of the pRF? Assume 1 standard deviation.
    n_std = 1
    left_extent = prf_params[:,0] - prf_params[:,2]*n_std
    right_extent = prf_params[:,0] + prf_params[:,2]*n_std
    top_extent = prf_params[:,1] + prf_params[:,2]*n_std
    bottom_extent = prf_params[:,1] - prf_params[:,2]*n_std

    out_of_bounds = (left_extent > 0.5) | (right_extent < -0.5) | (top_extent < -0.5) | (bottom_extent > 0.5)

    prf_params = prf_params[~out_of_bounds,:]

    return prf_params


def gauss_2d(center, sd, patch_size, orient_deg=0, aperture=1.0, dtype=np.float32):
    """
     Making a little gaussian blob. Can be elongated in one direction relative to other.
     [sd] is the x and y standard devs, respectively. 
     center and size are scaled according to the patch size, so that the blob always 
     has the same center/size relative to image even when patch size is different.
     aperture defines the number of arbitrary "units" occupied by the whole image
     units occupied by each pixel = aperture/patch_size.
        
     """
    if (not hasattr(sd,'__len__')) or len(sd)==1:
        sd = np.array([sd, sd])
        
    aspect_ratio = sd[0] / sd[1]
    orient_rad = orient_deg/180*np.pi
    
    # first meshgrid over image space
    x,y = np.meshgrid(np.linspace(-aperture/2, aperture/2, patch_size), \
                      np.linspace(-aperture/2, aperture/2, patch_size))
    
    new_center = copy.deepcopy(center) # make sure we don't edit the input value by accident!
    new_center[1] = (-1)*new_center[1] # negate the y coord so the grid matches w my other code
    
    x_centered = x-new_center[0]
    y_centered = y-new_center[1]
    
    # rotate the axes to match desired orientation (if orient=0, this is just regular x and y)
    x_prime = x_centered * np.cos(orient_rad) + y_centered * np.sin(orient_rad)
    y_prime = y_centered * np.cos(orient_rad) - x_centered * np.sin(orient_rad)

    # make my gaussian w the desired size/eccentricity
    gauss = np.exp(-((x_prime)**2 + aspect_ratio**2 * (y_prime)**2)/(2*sd[0]**2))
    
    # normalize so it will sum to 1
    gauss = gauss/np.sum(gauss)
    
    gauss = gauss.astype(dtype)
    
    return gauss

def get_prf_mask(center, sd, patch_size, zscore_plusminus=2):
    
    """
    Get boolean mask for each pRF (region +/- n sds from center)
    zscore_plusminus determines how many stdevs out.
    """
    
    # cutoff of 0.14 approximates +/-2 SDs
    cutoff_height = np.round(zscore_to_pdfheight(zscore_plusminus), 2)
    
    if np.all(np.abs(center)<0.50):
        
        # if the center of the pRF is within the image region, 
        # then can get max value without padding.
        prf = gauss_2d(center, sd, patch_size, aperture=1.0)
       
        prf_mask = prf/np.max(prf)>cutoff_height
        
    else:
        
        # otherwise need to pad array a little so that the center 
        # (max) will be included in the image.
        grid_space = 1.0/(patch_size-1)
        spaces_pad = int(np.ceil(0.5/grid_space))
        padded_aperture = 1.0+grid_space*spaces_pad*2
        padded_size = patch_size+spaces_pad*2
        prf_padded = gauss_2d(center, sd, patch_size=padded_size, \
                                 aperture=padded_aperture)
        
        prf_mask_padded = prf_padded/np.max(prf_padded)>cutoff_height
        
        # now un-pad it back to original size.
        prf_mask = prf_mask_padded[spaces_pad:spaces_pad+patch_size, \
                                   spaces_pad:spaces_pad+patch_size]
        
    return prf_mask

def zscore_to_pdfheight(z_target, normalized=True):
    
    assert(np.abs(z_target)<5)
    
    x = np.linspace(-5,5,1000)

    y = scipy.stats.norm.pdf(x)
    if normalized:
        y /= np.max(y)

    nearest_x_ind = np.argmin(np.abs(x-z_target))
    h_target = y[nearest_x_ind]
    
    return h_target

def pol_to_cart(angle_deg, eccen_deg):
    """
    Convert from polar angle coordinates (angle, eccentricity)
    to cartesian coordinates (x,y)
    Inputs and outputs in units of degrees.
    """
    angle_rad = angle_deg*np.pi/180
    x_deg = eccen_deg*np.cos(angle_rad)
    y_deg = eccen_deg*np.sin(angle_rad)
    
    return x_deg, y_deg

def cart_to_pol(x_deg, y_deg):
    """
    Convert from cartesian coordinates (x,y)
    to polar angle coordinates (angle, eccentricity)
    Inputs and outputs in units of degrees.
    """
    x_rad = x_deg/180*np.pi
    y_rad = y_deg/180*np.pi
    angle_rad = np.mod(np.arctan2(y_rad,x_rad), 2*np.pi)
    angle_deg = angle_rad*180/np.pi
    
    eccen_deg = np.sqrt(x_deg**2+y_deg**2)
    
    return angle_deg, eccen_deg

def get_prf_grid(grid_name = 'default-log-polar'):

    """
    # This is my default pRF grid, as used in previous papers.
    # in older code, this is called "grid 5"
    # in newer code, just called "default-log-polar"
    # it's a log-polar grid (centers and sizes are in polar coordinates, log-spaced)
    # see methods of:
    # https://doi.org/10.1523/JNEUROSCI.1822-22.2023
    # https://doi.org/10.1167/jov.23.4.8
    
    # returns: models, [n_candidates x 3]
    # where columns are: [x, y, sigma]
    # Units are all relative to image aperture size (1.0 units)
    # Assume that the image has a total size of 1.0 units.
    # X and Y coordinates of the image span from [-0.5, 0.5]
    # To convert to degrees, multiply these values by degrees of display (8.4 degrees for NSD expts)

    # Note that some centers do go outside the image region (like 0.7). 
    # Those are included because they partially overlap the image extent, even though center is outside.
    """
    
    models = make_log_polar_grid(sigma_range=[0.02, 1], n_sigma_steps=10, \
                              eccen_range=[0, 7/8.4], n_eccen_steps=10, n_angle_steps=16)  
    
    return models, grid_name


def discretize_nsd_mapping_prfs(subject, grid_name = 'default-log-polar'):

    """
    Converting pRF definitions from the pRF mapping task (which are continous)
    into the closest parameters from a grid of pRFs.
    Can be used for fitting pRF-constrained encoding models
    """

    # first creating the discrete grid of candidate pRFs
    grid_params, grid_name = get_prf_grid(grid_name = grid_name)
    x_grid, y_grid, sigma_grid = grid_params.T

    # where my preprocessed NSD files live
    # this is where we have the pre-computed pRF estimates for each voxel, for each subject
    nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
    rois_folder = os.path.join(nsd_path, 'rois')

    # load the pRF estimates (see "proc_rois.py")
    fn = os.path.join(rois_folder, 'S%d_prf_params.npy'%subject)
    prf_info = np.load(fn, allow_pickle=True).item()

    # units of degrees
    a = prf_info['angle']
    e = prf_info['eccen']
    s = prf_info['size']
    
    # Converting them into [x, y] coordinates, from polar
    # The units here are degrees visual angle
    x_mapping, y_mapping = pol_to_cart(a,e)
    
    # this step is taking out any extreme values, way outside the image range.
    # 7 is the max extent of the pRF grid in x,y (0.8333 * 8.4)
    x_mapping = np.minimum(np.maximum(x_mapping, -7), 7)
    y_mapping = np.minimum(np.maximum(y_mapping, -7), 7)
    
    # S = pRF size, sigma of the Gaussian function. Degrees visual angle.
    # again taking out any huge values here. 8.4 is max degrees of pRF grid.
    s_mapping = np.minimum(s, 8.4)

    # now converting these to same coords as my pRF grid.
    # this is relative to image size, assuming image has size of 1.0 
    x_image_coords = x_mapping / 8.4
    y_image_coords = y_mapping / 8.4
    s_image_coords = s_mapping / 8.4

    n_vox = len(x_image_coords)

    prf_grid_inds = np.zeros((n_vox,1),dtype=int)
    
    for vv in range(n_vox):

        if np.any(np.isnan([x_image_coords[vv], y_image_coords[vv], s_image_coords[vv]])):
            print('skipping one vox with nan value')
            prf_grid_inds[vv] = -1000
            continue
    
        # first find the [x,y] coordinate closest to this pRF center (in my grid)
        distances_xy = np.sqrt((x_image_coords[vv]-x_grid)**2 + (y_image_coords[vv]-y_grid)**2)
    
        # should be multiple possible values here, for the different sizes
        closest_xy_inds = np.where(distances_xy==np.min(distances_xy))[0]
    
        # then find which size is closest to mapping task estimate
        distances_size = np.abs(s_image_coords[vv] - sigma_grid[closest_xy_inds])
    
        closest_ind = closest_xy_inds[np.argmin(distances_size)]
    
        prf_grid_inds[vv] = closest_ind.astype(int)

    save_folder = '/lab_data/hendersonlab/features/NSD_prfmodel/voxel_prfs/mapping_prfs'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # save these results as npy file
    save_filename = os.path.join(save_folder, 'NSD_S%d_mapping_prfs_%s.npy'%(subject, grid_name))
    print(save_filename)
    np.save(save_filename, \
            {'voxel_prf_grid_inds': prf_grid_inds,\
             'prf_grid_params': grid_params},\
             allow_pickle=True)

    # also save the pRF grid itself, for convenience
    save_filename = os.path.join(save_folder, 'Params_pRF_grid_%s.csv'%(grid_name))
    print(save_filename)
    df = pd.DataFrame(np.concatenate([grid_params, grid_params * 8.4], axis=1), \
                                 columns = ['x','y','sigma','x_deg','y_deg','sigma_deg'])
    df = df.round(4)
    df.to_csv(save_filename)

    print('Done')


def get_prf_stack(n_pix, which_prf_grid = 'default-log-polar', dtype=np.float32):

    # Create a stack of candidate pRFs
    # [x, y, num pRFs]
    
    models, grid_name = get_prf_grid(grid_name = which_prf_grid)   
    n_prfs = models.shape[0]

    prf_stack = []
    for pi in np.arange(n_prfs):
    
        x, y, sigma = models[pi]
        aperture = 1.0
        g = gauss_2d(center=[x,y], sd=sigma, \
                                 patch_size=n_pix, aperture=aperture, dtype=dtype)
    
        prf_stack += [g]
    
    prf_stack = np.stack(prf_stack, axis=2)
    
    return prf_stack


def get_prfs_from_fwrf_fit():
    """
    This is used to gather pRF parameters from a previously fit FWRF model.
    These are fitted saved models from our gabor model paper: https://doi.org/10.1167/jov.23.4.8
    This function will load the saved encoding model fits, pull out the pRF params, 
    and save it as a new file that is easier to access.
    """

    save_path = '/lab_data/hendersonlab/features/NSD_prfmodel/voxel_prfs/'
    
    model_fits_path = '/user_data/mmhender/image_stats_gabor/model_fits/'
    
    alexnet_fit_paths = ['S01/alexnet_all_conv_pca/Apr-01-2022_1317_39/all_fit_params.npy', \
                     'S02/alexnet_all_conv_pca/Apr-02-2022_2104_46/all_fit_params.npy', \
                     'S03/alexnet_all_conv_pca/Apr-04-2022_0349_08/all_fit_params.npy',  \
                     'S04/alexnet_all_conv_pca/Apr-05-2022_1052_06/all_fit_params.npy', \
                     'S05/alexnet_all_conv_pca/Apr-07-2022_1401_20/all_fit_params.npy', \
                     'S06/alexnet_all_conv_pca/Apr-10-2022_1650_18/all_fit_params.npy', \
                     'S07/alexnet_all_conv_pca/Apr-11-2022_2255_10/all_fit_params.npy', \
                     'S08/alexnet_all_conv_pca/Apr-13-2022_0045_36/all_fit_params.npy']
    alexnet_fit_paths = [os.path.join(model_fits_path, aa) for aa in alexnet_fit_paths]
    
    for si, fn in enumerate(alexnet_fit_paths):

        print(fn)
        out = np.load(fn, allow_pickle=True).item()
        prf_grid_inds = out['best_params'][5][:,0]
        
        save_subfolder = os.path.join(save_path, 'alexnet_fwrf_prfs')
        if not os.path.exists(save_subfolder):
            os.makedirs(save_subfolder)
        ss = si+1
        save_fn = os.path.join(save_subfolder, 'NSD_S%d_alexnet_fwrf_prfs_default-log-polar.npy'%ss)
        print(save_fn)
        np.save(save_fn, {'voxel_prf_grid_inds': prf_grid_inds, \
                         'original_fit_filename': fn, \
                         'prf_grid_params': out['models'], \
                         'which_prf_grid': out['which_prf_grid']}, allow_pickle=True)
        
    gabor_fit_paths = ['S01/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-04-2022_1525_10/all_fit_params.npy', \
                     'S02/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-04-2022_1759_56/all_fit_params.npy', \
                     'S03/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-04-2022_2035_29/all_fit_params.npy', \
                     'S04/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-05-2022_0511_36/all_fit_params.npy', \
                     'S05/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-05-2022_0718_44/all_fit_params.npy', \
                     'S06/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-05-2022_0947_32/all_fit_params.npy', \
                     'S07/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-05-2022_1224_52/all_fit_params.npy', \
                     'S08/gabor_solo_ridge_12ori_8sf_fit_pRFs/Apr-05-2022_1437_12/all_fit_params.npy']
    gabor_fit_paths = [os.path.join(model_fits_path, aa) for aa in gabor_fit_paths]

    for si, fn in enumerate(gabor_fit_paths):

        print(fn)
        out = np.load(fn, allow_pickle=True).item()
        prf_grid_inds = out['best_params'][5][:,0]
        
        save_subfolder = os.path.join(save_path, 'gabor_fwrf_prfs')
        if not os.path.exists(save_subfolder):
            os.makedirs(save_subfolder)
        ss = si+1
        save_fn = os.path.join(save_subfolder, 'NSD_S%d_gabor_fwrf_prfs_default-log-polar.npy'%ss)
        print(save_fn)
        np.save(save_fn, {'voxel_prf_grid_inds': prf_grid_inds, \
                         'original_fit_filename': fn, \
                         'prf_grid_params': out['models'], \
                         'which_prf_grid': out['which_prf_grid']}, allow_pickle=True)
        
    
    
