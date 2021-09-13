"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
import os
import nibabel as nib
import util.cmr_dataloader as cmr
import util.cmr_transform as cmr_tran
from torch.utils.data import DataLoader
from torchvision.transforms import Compose



class CmrDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.add_argument('--image_dir_A', type=str, default='/data/sina/datasets/seb/SA_files_processed/', help='path to the cmr images')
        parser.add_argument('--image_dir_B', type=str, default='/data/sina/dataset/seb/mms2_processed/Philips/', help='path to the cmr images')
        parser.add_argument('--target_res', type=float, default=2, help='resample images to this resolution')
        parser.add_argument('--target_crop', type=int, default=256, help='center crop images to this size')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser
    

        #Sina starts
    def prepare_set(self, opt):
        """
        To prepare and get the list of files
        """
        

        if opt.valid_data:
            folder = 'ImageValid'
        else:
            folder = 'Image'
        Image_list_A = sorted(os.listdir(os.path.join(opt.image_dir_A, folder)))
        Image_list_B = sorted(os.listdir(os.path.join(opt.image_dir_B, folder)))
        assert len(Image_list_A) != 0 , 'list of images doesnt exist'
        assert len(Image_list_B) != 0 , 'list of images doesnt exist'

        
        filename_pairs_A = []
        filename_pairs_B = [] 
        for i in range(len(Image_list_A)):
            filename_pairs_A += [(os.path.join(opt.image_dir_A, folder, Image_list_A[i]), os.path.join(opt.image_dir_A, folder, Image_list_A[i]))]
        for i in range(len(Image_list_B)):
            filename_pairs_B += [(os.path.join(opt.image_dir_B, folder, Image_list_B[i]), os.path.join(opt.image_dir_B, folder, Image_list_B[i]))]

        self.Image_list_A = Image_list_A
        self.Image_list_B = Image_list_B
        self.filename_pairs_A = filename_pairs_A
        self.filename_pairs_B = filename_pairs_B
    

    def __init__(self, opt):
        self.opt = opt
        self.prepare_set(opt)
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """


        if opt.phase == 'train':
            train_transforms = Compose([
                # cmr_tran.Resample(self.opt.target_res,self.opt.target_res), #1.33
                # cmr_tran.CenterCrop2D((self.opt.target_crop,self.opt.target_crop)),
                # cmr_tran.RandomRotation(degrees=90),
                # cmr_tran.RandomRotation(p=0.5),
                cmr_tran.ToTensor(),
                # cmr_tran.SimpleIntensityShift(),
                cmr_tran.PercentileBasedRescaling(out_min_max=(-1,1), percentiles=(1,99)),  #TODO: make sure the normalization is performed on the volume data not slice-by-slice
                # cmr_tran.RandomElasticTorchio(num_control_points  = (8, 8, 4), max_displacement  = (20, 20, 0), p=0.5),
                # cmr_tran.ClipScaleRange(min_intensity= 0, max_intensity=1),
                # cmr_tran.ClipTanh(),
                # cmr_tran.ClipScaleRange(),
                # cmr_tran.ClipNormalize(min_intensity= 0, max_intensity=4000),
                # cmr_tran.ClipZscoreMinMax(min_intensity= 0, max_intensity=4000),
                # cmr_tran.RandomHorizontalFlip2D(p=0.50),
                # cmr_tran.RandomVerticalFlip2D(p=0.50),
                # cmr_tran.UpdateLabels(source=TR_CLASS_MAP_MMS_SRS, destination=TR_CLASS_MAP_MMS_DES)

            ])
        else:
            train_transforms = Compose([
                # cmr_tran.Resample(self.opt.target_res,self.opt.target_res), #1.33
                # cmr_tran.CenterCrop2D((self.opt.target_crop,self.opt.target_crop)),
                # cmr_tran.RandomDilation_label_only(kernel_shape ='elliptical', kernel_size = 3, iteration_range = (1,2) , p=0.5),
                # cmr_tran.RandomRotation(degrees=90),
                # cmr_tran.RandomRotation(p=0.5),
                cmr_tran.ToTensor(),
                # cmr_tran.SimpleIntensityShift(),
                cmr_tran.PercentileBasedRescaling(out_min_max=(-1,1), percentiles=(1,99)),  #TODO: make sure the normalization is performed on the volume data not slice-by-slice
                # cmr_tran.RandomElasticTorchio_label_only(num_control_points  = (8, 8, 4), max_displacement  = (14, 14, 1), p=1),
                
                # cmr_tran.RandomElasticTorchio(num_control_points  = (8, 8, 4), max_displacement  = (20, 20, 0), p=0.5),
                # cmr_tran.ClipScaleRange(min_intensity= 0, max_intensity=4000),
                # cmr_tran.ClipTanh(),
                # cmr_tran.ClipScaleRange(),
                # cmr_tran.RandomHorizontalFlip2D(p=0.5),
                # cmr_tran.RandomVerticalFlip2D(p=0.5),
                # cmr_tran.UpdateLabels(source=TR_CLASS_MAP_MMS_SRS, destination=TR_CLASS_MAP_MMS_DES)

            ])
        
        self.cmr_dataset_A = cmr.MRI2DSegmentationDataset(self.filename_pairs_A, transform = train_transforms, slice_axis=2, canonical = False)
        self.cmr_dataset_B = cmr.MRI2DSegmentationDataset(self.filename_pairs_B, transform = train_transforms, slice_axis=2, canonical = False)

        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        # self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)
        
        




    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        
        """

        input_image_A= self.cmr_dataset_A[index]
        input_image_B= self.cmr_dataset_B[index]
        path_A = input_image_A['filename']    # needs to be a string
        path_B = input_image_B['filename']    # needs to be a string
        data_A = input_image_A['input'].repeat(self.opt.output_nc, 1, 1)    # needs to be a tensor input domain
        data_B = input_image_B['input'].repeat(self.opt.output_nc, 1, 1)    # needs to be a tensor target domain
        input_dict = {
                    'A': data_A,
                    'B': data_B,
                    'A_paths': path_A,
                    'B_paths': path_B,
                    'index_A': input_image_A['index'], 'index_B': input_image_B['index'],
                    'segpair_slice_A': input_image_A['segpair_slice'], 'segpair_slice_B': input_image_B['segpair_slice']
                    }

        return input_dict

    def __len__(self):

        """Return the total number of images."""
        if self.cmr_dataset_A.__len__() >= self.cmr_dataset_B.__len__():
            data_size = self.cmr_dataset_B.__len__()
        else:
            data_size = self.cmr_dataset_A.__len__()
        return data_size
