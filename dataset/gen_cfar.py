"""This file converts the radar DREA cubes (from kradar) into a point cloud by performing the following:
    1- Projecting the RAD cube into th RD map by taking the sum of the magnitudes over the angle dimension
    2- Performing CFAR of choice on the RD Map 
    3- The detection point list is then projected back to the RAD cube to extract the power, 
    velocity and cartesian x,y,z of each point (Polar to cartesian transformation needed)
    4- Peak search 
    It also plots some of the spectra with threshold shown in graph for validation 
"""
import numpy as np
import os 
from glob import glob
import numpy as np
from pypcd4 import PointCloud
from PIL import Image
from pathlib import Path
import scipy.io as sio
import matplotlib.pyplot as plt



def plotSpec(spec:np.ndarray,name:str):
    #pick the first spectrum
    im = Image.fromarray(10*np.log((spec/(spec.mean())*255)+1e-15))

    im = im.convert('RGB')
    im.save(name+"_RD_cfar.jpeg")
    pass
    
def get_list_of_seq_files(base_path:str, sensor_name:str):
    path = base_path+"/*/"
    path_per_seq_list = []
    dataset_seq_paths = sorted(glob(path), reverse=True)
    dataset_seq_paths = [pth+sensor_name for pth in dataset_seq_paths]
    for seq in dataset_seq_paths:
        print(seq)
        files = sorted(glob(seq+"/*"), reverse=True)
        path_per_seq_list.append(files)

        seq_name =  os.path.basename(os.path.dirname(seq))
        save_path_dir = base_path +f"/{seq_name}/radar_pointcloud3p0/"

        if not Path.exists(Path(save_path_dir)):
            os.makedirs(save_path_dir)

        for file in files:
            if os.path.getsize(file) > 290200000:
                file_name_noext = Path(file).stem
                save_file = save_path_dir+"pointcloud"+file_name_noext[-6:]+".pcd"
                if not Path.exists(Path(save_file)):
                    try:
                        spec2d, packedpoints,idcs = cfar1d(file)
                        # print(file)
                        pc = create_pc(packedpoints)
                        pc.save_pcd(save_file,compression='binary') 

                    except:
                        print(f"{file} is corrupt, not converted to pointcloud")
                        
    return path_per_seq_list

def create_pc(xyz_velPow):
    xyz_vps = []
    md = {'version': .7,
        'fields': ['x', 'y', 'z','vr', 'power', 'snr'],
        'size': [4, 4, 4, 4, 4, 4],
        'type': ['F', 'F', 'F','F', 'F', 'F'],
        'count': [1, 1, 1, 1, 1, 1],
        'width': xyz_velPow.shape[0],
        'height': 1,
        'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        'points': xyz_velPow.shape[0],
        'data': 'binary'}
    #Pick only x,y,z,vr,pow,snr
    for idx in range(xyz_velPow.shape[0]):
        x = xyz_velPow[idx,0]
        y = xyz_velPow[idx,1]
        z = xyz_velPow[idx,2]
        vr = xyz_velPow[idx,3]
        pow = xyz_velPow[idx,4]
        snr = 1.
        xyz_vps.append([x,y,z,vr,pow,snr])
    xyz_vps = np.array(xyz_vps,dtype=np.float32)
    pc_data = xyz_vps.view(np.dtype([('x', np.float32),
                                    ('y', np.float32),
                                    ('z', np.float32),
                                    ('vr', np.float32),
                                    ('power', np.float32),
                                    ('snr', np.float32),
                                    ])).squeeze()
    # print(len(pc_data))
    # print(pc_data.shape)
    pc = PointCloud(md, pc_data)
    
    return pc


def plot_BEV(idcs,shape):
    img = np.zeros(shape)
    img[idcs[:,0],idcs[:,1]] = 255
    im = Image.fromarray(img)

    im = im.convert('RGB')
    im.save("kradar_cfar_RA.jpeg")
    
def plot_cartesian_full(points,shape):
    img = np.zeros(shape)
    img[np.floor(points[:,1]/ (0.4609)).astype(int),np.floor((90+points[:,0])/ (0.706)).astype(int)] = 255
    im = Image.fromarray(np.flip(img,axis=[0]))

    im = im.convert('RGB')
    im.save("kradar_cfar_cartesian.jpeg")
    
def plot_cartesian(points,roi_x=[-80.,0.4,80.],roi_y=[0.,0.4,100.]):
    shape = ((roi_y[2]-roi_y[0])/roi_y[1],(roi_x[2]-roi_x[0])/roi_x[1])
    img = np.zeros(np.round(shape).astype(int))
    inrange_x =np.where( np.logical_and((points[:,0])> roi_x[0] , (points[:,0])< roi_x[2]))
    inrange_y =np.where( np.logical_and((points[:,1])> roi_y[0] , (points[:,1])< roi_y[2]))
    valid_xy = np.intersect1d(inrange_y[0],inrange_x[0])
    img[np.floor((points[valid_xy,1])/0.4).astype(int),np.floor((80+points[valid_xy,0])/0.4).astype(int)] = 255
    im = Image.fromarray(np.flip(img,axis=[0]))

    im = im.convert('RGB')
    im.save("kradar_cfar_cartesian.jpeg")
    
    
    
def cfar1d(spec_path, guardcells=4, traincells=32, scaling_factor=3):
    """Inputs:
            spec_path: path to a 4D DREA spectrum in mat extension,
            guardcells: number of guard cells to the left and right of target cell, default is 4, 
            traincells: number of training cellx, if None, the size will be the doppler size, default is None.
            scaling_factor: scaling factor in watts. default 10
        Outputs: 
            spectrum2d_out: 2D RD spectrum, with values above peak retained, otherwise zeros:  nRange x nDoppler
            xyz_velPow: A 2D np array of packed point cloud with 5 features after CFAR:  nPoints x 5
            idcs: Range azimuth indeces of the peaks
    """
    ### FOR RADDet###
    doppler_resol = 0.06
    range_resol = 0.4628905882
    azim_resol = np.pi/180.
    elev_resol = np.pi/180.
    
    
    spectrum4d = sio.loadmat(spec_path)["arrDREA"]
    #if not specified, the training cells will be the whole doppler dimension (dummy threshold)
    traincells = traincells if traincells else spectrum4d.shape[0]//2
    spectrum2d = np.absolute(spectrum4d).mean(axis = (2,3)).reshape(spectrum4d.shape[0],spectrum4d.shape[1]).T
    plotSpec(spectrum2d,"before")
    gc = guardcells
    tc = traincells
    # print(spectrum2d.shape)
    threshold_levels = np.zeros(spectrum2d.shape)
    #for each row in the RD, 
    for idx_rng in range(spectrum2d.shape[0]):
        num_velocity_cell = spectrum2d.shape[1]
        for idx_vel in range(num_velocity_cell):
            # get the indices of the trining cells in range dim
            np.arange((idx_vel-gc-tc),(idx_vel-gc-1))
            tc_idx = np.array((np.arange(idx_vel-gc-tc,idx_vel-gc-1),np.arange(idx_vel+gc+1,idx_vel+gc+tc)))
            #for points outseide the edges, circular indeces used
            # print(tc_idx[tc_idx < 0])
            # print(tc_idx[tc_idx>num_velocity_cell])
            tc_idx[tc_idx < 0] = tc_idx[tc_idx < 0]+ num_velocity_cell; 
            tc_idx[tc_idx >=num_velocity_cell] = tc_idx[tc_idx >= num_velocity_cell]- num_velocity_cell; 
            # print(tc_idx)
            
            # calculate the mean of the training cells 
            avg =  np.mean(spectrum2d[idx_rng,tc_idx])
            threshold_levels[idx_rng,idx_vel] = avg *scaling_factor
    spectrum2d[spectrum2d<threshold_levels]=0
    # print((spectrum2d>threshold_levels))
    range_vel_idcs =  np.transpose((spectrum2d>threshold_levels).nonzero())
    range_idcs = range_vel_idcs[:,0]
    vel_idcs =  range_vel_idcs[:,1]
    ranges = range_idcs * range_resol
    doppler_fft_shift = spectrum4d.shape[0]//2
    vels = (vel_idcs - doppler_fft_shift)* doppler_resol 

    elev_azim = np.argmax(np.abs(spectrum4d)[vel_idcs,range_idcs].reshape(-1,107*37),axis = 1)
    elev_azim_idcs = np.unravel_index(elev_azim,spectrum4d.shape[2:4])
    # thetas = thetas_idcs * angle_resol +abs(np.pi- angle_resol *256)/2
    elev_idcs = elev_azim_idcs[0]
    azim_idcs = elev_azim_idcs[1]
    
    # thetas = np.arcsin(np.array([((j * (2*np.pi/256) - np.pi) / (2*np.pi*0.5*77/76.8)) for j in thetas_idcs]))
    elevs = (elev_idcs-18)*elev_resol
    azim = np.pi/2.-(azim_idcs-53)*azim_resol  
    # azim = (azim_idcs-spectrum4d.shape[3]//2)*azim_resol  +(107.-180.)/2. *np.pi
    pows = 10*np.log10(np.abs(spectrum4d[vel_idcs,range_idcs,elev_idcs,azim_idcs])**2)
    idcs = np.column_stack((range_idcs, azim_idcs))
    x = -(ranges) * np.cos(azim) * np.cos(elevs) 
    y = (ranges) * np.sin(azim) * np.cos(elevs) 
    z = (ranges) * np.sin(elevs) 
    xyz_velPow = np.vstack((x,y,z,vels,pows)).T

    return spectrum2d, xyz_velPow,idcs

def plot_RA(path):
        #pick the first spectrum
    spectrum4d = sio.loadmat(path)["arrDREA"]
    spec2d = np.abs(spectrum4d).sum(axis = (0,2)).reshape((256,107))
    im = np.flip(10*np.log((spec2d/(spec2d.mean())*255)+1e-15),axis=[0,1])

    plt.imshow(im, cmap='jet')
    # plt.colorbar()
    plt.savefig("kradar_RA_Spectrum.jpeg",bbox_inches='tight')
if __name__=="__main__":

    dataset_path_train = "XXX/k-radar/full/from_hdd"
    sensor_name = "radar_tesseract"

    train_dataset_seq_paths = get_list_of_seq_files(dataset_path_train,sensor_name)
    # file_path = dataset_path_train+"/31/radar_tesseract/tesseract_00223.mat"
    # spec2d, packedpoints,idcs = cfar1d(file_path)
    # # # print(idcs)
    # plot_BEV(idcs,[256,107])
    # plot_cartesian(packedpoints)
    # plotSpec(spec2d,"1tesseract_00004")
    # plot_RA(file_path)
    pass
    