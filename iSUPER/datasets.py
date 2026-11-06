import glob
import numpy as np
import torch
import pandas as pd
from math import pi
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

## Processes data for NP
## No common aux for now
## Returns x_data and y_data
## Make sure aq_df and s_code is sorted
## if c_set = [], then all the sensors are used
## Output Lat, lon, humidity, temperature
def aq_processing_np (aq_df, c_set, s_code, diff_aux = []):

    if (len(c_set) == 0):
        c_set = list(range(aq_df.shape[1]))
        
    c_df = aq_df.iloc[:, c_set]

    ## Stacked covariate array (temp, hum) into numpy array with size time_points * num_sensors * 4 
    ## TODO : 1. Add temperature and humidity
    ##        2. Think of ways to make training set/ test set

    context_s_code = [s_code[i] for i in c_set]
    lats = np.array([item[1] for item in context_s_code])  # shape: (20,)
    lons = np.array([item[2] for item in context_s_code])  # shape: (20,)

    # Define M (number of rows you want)
    num_rows = aq_df.shape[0]

    # Create a M x num_rows array by repeating the values as rows
    lat_arr = np.tile(lats, (num_rows, 1))  
    lon_arr = np.tile(lons, (num_rows, 1))  

    aa_list = [lat_arr, lon_arr]

    ## Process other covariates (temperature, humidity)
    for i in range(len(diff_aux)):
        aa = diff_aux[i].iloc[:, c_set].to_numpy()
        aa_list.append(aa)

    ## final result : fin_aa (shape : [Number of datapoints, number of context points ,data dimension])
    ## final result : fin_y (shape : [Number of datapoints, number of context points ,data dimension])
    fin_aa = np.stack(aa_list, axis=-1)
    fin_y = np.expand_dims(c_df, axis=2)
    return fin_aa, fin_y

## Log normalization for PM2.5
## Then apply min-max normalization for all data
def transform_np_1(x_data, y_data, min_arr = [], max_arr = []):
    # Log normalization
    yy = np.log(y_data)

    if ((len(min_arr) == 0 ) or (len(max_arr) == 0)):
        # Compute min and max along the third dimension (axis=2)
        min_x = np.min(x_data, axis = (0,1),keepdims = True)
        min_x = np.tile(min_x, (x_data.shape[0], x_data.shape[1], 1))
        max_x = np.max(x_data, axis = (0,1),keepdims = True)
        max_x = np.tile(max_x, (x_data.shape[0], x_data.shape[1], 1))
        #min_y = yy.min()
        #max_y = yy.max()

    else : 
        min_x = np.array(min_arr)
        max_x = np.array(max_arr)

    # Avoid division by zero (in case min == max)
    div_x = max_x - min_x
    div_x[div_x == 0] = 1e-8  # Or choose a small epsilon
    #div_y = max_y - min_y
    #if(div_y == 0):
    #    div_y = 1e-8

    # Min-max normalize along axis=2
    normalized_x = (x_data - min_x) / div_x
    #normalized_y = (yy - min_y) / div_y
    normalized_y = yy

    return normalized_x, normalized_y
    

## Input : training set(AQ) dataframe, test set(AQ) dataframe
## Output : flattened latitude and longitude np array
def coordinate_processing(air_df, tar_set, sens_c):
    lat_list = []
    lon_list = []
        
    for i in tar_set:
        lat_val = sens_c[i][1]
        lon_val = sens_c[i][2]

        lat_list.append(lat_val)
        lon_list.append(lon_val)
        
    ## Copy and make into 2d array
    lat_np = np.tile(lat_list, (air_df.shape[0], 1)).flatten(order='F').reshape(-1, 1)
    lon_np = np.tile(lon_list, (air_df.shape[0], 1)).flatten(order='F').reshape(-1, 1)
    return lat_np, lon_np

## Input : aux_df(a dataframe where row : time / column : sensor code)
## Output : flattened covariate numpy
def gen_aux_processing(aux_df, tar_set):
    ret_np = aux_df.iloc[:, tar_set].to_numpy().flatten(order='F').reshape(-1, 1)
    return ret_np

## c_set and t_set should be sorted
## Common aux (covariates applied the same way) : wind speed & direction (If the same sensors are used)
## Different aux (covariates applied in a different way) : location, wind speed, wind direction
## Output : Processed x_data (2d array) and y_data (1d array)
def aq_processing(aq_df, c_set, t_set, s_code, comm_aux = [], diff_aux = []):

    ## Prepare context set data for aq_data
    c_df = aq_df.iloc[:, c_set]
    t_df = aq_df.iloc[:, t_set]

    ## Prepare common_aux
    if(len(comm_aux) > 0):
        c_aux = comm_aux[0]
    for i in range(1,len(comm_aux)):
        c_aux = pd.concat([c_aux, comm_aux[i]], axis = 1)
        
    ## Concat contex aq_data + common_aux
    if (len(comm_aux) > 0):
        c_df_merge1 = pd.concat([c_df, c_aux], axis = 1).to_numpy()
    else :
        c_df_merge1 = c_df.to_numpy()

    ## Reshape in the form needed
    x_data = np.tile(c_df_merge1, (len(t_set), 1))
    
    ## Prepare Label
    yy = t_df.to_numpy().flatten(order='F')
    
    ## Make test set for each label
    ## make additional function
    ## 1. Location (Coordinates)
    lat, lon = coordinate_processing(aq_df, t_set, s_code)
    ## 2. Temperature / Humidity
    ## 3. Wind speed/direction (Future)
    other_aux = []
    other_aux.append(lat)
    other_aux.append(lon)
    for i in range(len(diff_aux)):
        other_aux.append(gen_aux_processing(diff_aux[i], t_set))
    ## 4. Traffic / Land Use

    ## Make final x data
    for i in range(len(other_aux)):
        x_data = np.concatenate((x_data, other_aux[i]), axis = 1)

    return x_data, yy

"""
#print("Predefined training / test set")
        tr_set_2 = sorted(trset)
        tt_set_2 = sorted(ttset)

        for i in tr_set_2:
            trlat.append(coor[i][1])
            trlon.append(coor[i][2])
            #training.append(dtset[i])

        for j in tt_set_2:
            ttlat.append(coor[j][1])
            ttlon.append(coor[j][2])
            #testing.append(dtset[j])

        fin_trset = tr_set_2.copy()
        fin_ttset = tt_set_2.copy()

        if (dtset.ndim == 1):
            training = dtset[tr_set_2]
            testing = dtset[tt_set_2]
        else :
            training = dtset[:, tr_set_2]
            testing = dtset[:, tt_set_2]
            
    return trlat, trlon, ttlat, ttlon, training, testing, fin_trset, fin_ttset
"""
## Applies Linear_Scaling (min_max_normalization) or z-norm scaling
## Input : numpy array of data
## If no scaling should be applied, set min  = 0 and max = 1
## For min-max scaling apply minimum and maximum - minimum
## For z_normalization appply mean and stdev
def custom_norm(the_np, min, range):
    #min_np = np.array(min)
    #range_np = np.array(range)
    ret_np = np.divide(the_np - min, range)
    return ret_np

def min_max_norm(the_np):
    min_arr = np.min(the_np, axis = 0)
    range_arr = np.max(the_np, axis = 0) - min_arr
    ret_arr = np.divide(the_np - min_arr, range_arr)
    return ret_arr

def z_norm(the_np):
    mean_arr = np.mean(the_np, axis = 0)
    std_arr = np.std(the_np, axis = 0)
    ret = np.divide(the_np - mean_arr, std_arr)
    return ret

## Input : x_data as numpy and y_data as numpy and the number of sensors 
# (the first num_sens datapoints in a row are the x_data)
## Transform 
## Apply log to the PM2.5 data 
## Apply z-norm on entire data (Manually found - Mean : 1.96, std : 0.67)
def transform_1(x_np, y_np, num_sens):
    tr_x = np.copy(x_np)
    ## Apply Log normalization
    tr_x[:, :num_sens] = np.log(tr_x[:, :num_sens])
    ## Apply z-normalization
    minus_arr = np.full(x_np.shape[1], 1.96) 
    div_arr = np.full(x_np.shape[1], 0.67)
    minus_arr[num_sens:] = np.mean(tr_x[:, num_sens:], axis = 0)
    div_arr[num_sens:] = np.std(tr_x[:, num_sens:], axis = 0)
    tr_ret = custom_norm(tr_x, minus_arr, div_arr)
    tr_y = np.log(y_np)
    tr_y = (tr_y - 1.96)/0.67
    return tr_ret, tr_y
    
class CSVDataset(Dataset):
    def __init__(self, aq_path, sens_code, context_set, target_set, diff_aux_path = [], common_aux_path = [], transform = transform_1, num_points = 24):
        """
        Args:
            aq_path : A string with Path to the CSV file. (One PM2.5 csv file )
            diff_aux_path : A list of paths that are applied differently for each sensor. Should be same format as aq_path files (ex : temperature, humidity )
            common_aux_path : A list of paths to covariates that are applied in the same way for all sensors (ex: wind speed, wind direction)
            sens_code : a list where each entries are [sensor number, latitude, longitude]
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        ## Reads all the data from csv files
        self.aq_data = pd.read_csv(aq_path)
        self.aux1 = [] ## different aux
        for i in range(len(diff_aux_path)):
            self.aux1.append(pd.read_csv(diff_aux_path[i]))
        self.aux2 = [] ## common aux
        for i in range(len(common_aux_path)):
            self.aux2.append(pd.read_csv(common_aux_path[i]))
            
        self.transform = transform
        
        x_dt, y_dt = aq_processing(self.aq_data, context_set, target_set, sens_code, comm_aux = self.aux2, diff_aux = self.aux1)
        #self.data = (x_dt, y_dt)
        self.x_data = x_dt
        self.y_data = y_dt

        ## Transform the data
        if (self.transform):
            self.trans_x, self.trans_y = self.transform(x_dt, y_dt)
        

    def __len__(self):
        """Returns the number of datapoints."""
        return len(self.y_data.size)

    
    def __getitem__(self, idx):
        """Retrieves a single data point."""
        #sample = self.features.iloc[idx].values.astype('float32')
        sample = self.x_data[idx]
        
        #if self.targets is not None:
        #    target = self.targets.iloc[idx].astype('float32')
        #else:
        #    target = None
        
        if self.transform:
           return (self.trans_x[idx], self.trans_y[idx])
        
        #return (sample, target) if target is not None else sample
        return (sample, self.y_data[idx])

class NPAirDataset(Dataset):
    def __init__(self, aq_path, sens_code, context_set, diff_aux_path = [], common_aux_path = [], transform = transform_np_1):
        """
        Args:
            aq_path : A string with Path to the CSV file. (One PM2.5 csv file )
            diff_aux_path : A list of paths that are applied differently for each sensor. Should be same format as aq_path files (ex : temperature, humidity )
            common_aux_path : A list of paths to covariates that are applied in the same way for all sensors (ex: wind speed, wind direction)
            sens_code : a list where each entries are [sensor number, latitude, longitude]
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        ## Reads all the data from csv files
        self.aq_data = pd.read_csv(aq_path)
        self.aux1 = [] ## different aux
        for i in range(len(diff_aux_path)):
            self.aux1.append(pd.read_csv(diff_aux_path[i]))
        self.aux2 = [] ## common aux
        for i in range(len(common_aux_path)):
            self.aux2.append(pd.read_csv(common_aux_path[i]))
            
        self.transform = transform

        x_dt, y_dt = aq_processing_np(self.aq_data, context_set, sens_code, diff_aux = self.aux1)
        self.x_data = torch.Tensor(x_dt)
        self.y_data = torch.Tensor(y_dt)

        ## Transform the data
        if (self.transform):
            tr_x, tr_y = self.transform(x_dt, y_dt, 
                                        min_arr = [42.31, -71.17, 12.27, 9.68],
                                        max_arr = [42.35, -71.10, 95.26, 43.17] )
            self.trans_x = torch.Tensor(tr_x)
            self.trans_y = torch.Tensor(tr_y)
        

    def __len__(self):
        """Returns the number of datapoints."""
        return len(self.y_data)

    def x_shape(self):
        return self.x_data.shape

    def y_shape(self):
        return self.y_data.shape
    
    def __getitem__(self, idx):
        """Retrieves a single data point."""
        #sample = self.features.iloc[idx].values.astype('float32')
        sample = self.x_data[idx]
        
        #if self.targets is not None:
        #    target = self.targets.iloc[idx].astype('float32')
        #else:
        #    target = None
        
        if self.transform:
           return (self.trans_x[idx], self.trans_y[idx])
        
        #return (sample, target) if target is not None else sample
        return (sample, self.y_data[idx])


class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100, x_d = 1):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = x_d  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        x = torch.Tensor()
        for i in range(num_samples):
            for j in range(self.x_dim):
                # Sample random amplitude
                a = (a_max - a_min) * np.random.rand() + a_min
                # Sample random shift
                b = (b_max - b_min) * np.random.rand() + b_min
                # Shape (num_points, x_dim)
                x1 = torch.linspace(-pi, pi, num_points).unsqueeze(1)
                # Shape (num_points, y_dim)
                if (j > 0):
                    x = torch.concat((x,x1), dim = 1)
                else :
                    x = x1
            y = a * torch.sin(x.mean(dim = 1) - b).unsqueeze(1)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


def mnist(batch_size=16, size=28, path_to_data='../../mnist_data'):
    """MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def celeba(batch_size=16, size=32, crop=89, path_to_data='../celeba_data',
           shuffle=True):
    """CelebA dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image.

    crop : int
        Size of center crop. This crop happens *before* the resizing.

    path_to_data : string
        Path to CelebA data files.
    """
    transform = transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    celeba_data = CelebADataset(path_to_data,
                                transform=transform)
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=shuffle)
    return celeba_loader


class CelebADataset(Dataset):
    """CelebA dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        path_to_data : string
            Path to CelebA data files.

        subsample : int
            Only load every |subsample| number of images.

        transform : torchvision.transforms
            Torchvision transforms to be applied to each image.
        """
        self.img_paths = glob.glob(path_to_data + '/*.jpg')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = Image.open(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0
