"""
Fully connected pipeline. Specify FOVSize and NumImgs, and this will make a hdf5 file with the datasets
TrainYes, TrainNo, TestYes, TestNo, ValYes, and ValNo, each with the number of images equal to NumImgs,
and the image size FOVSize x FOVSize x 1 (grayscale).
The datasets TrainYes, TestYes, and ValYes will all contain craters.
All of the datasets will be normalized in one of two ways (or both ways), which can be changed in the make_norm.py file.
The hdf5 file will be saved under the path SavePath.

Then, another training set will be created using Full images (384x512 pixels) and normalized in the same way with Yes/No, Train/Val,Test splits.

Finally a third set can be made with just random images -- not for training but for predicting and finding real craters.
"""

import BlankGather
import CraterCreation
import make_norm
import os
from dask.distributed import Client
from datetime import datetime
from rich.traceback import install; install()

if __name__ == '__main__':
    t0 = datetime.now()
    client = Client(processes=True)

    FOVSize = (150,150)
    NumImgs = 10000
    Dir = os.path.join("..", "Data")
    Name = 'Craters.hdf5'
    SavePath = os.path.join(Dir, Name)
    InputFileName = 'GlobbedFile_Aug6.txt'

    ### Gather blank backgrounds
    # False is for whether to do full size FOVs or not.
    BlankGather.blanks_do(FOVSize, False, NumImgs, Dir, InputFileName, InputFileName, SavePath, client)
    print('blanks created \n')

    ### Insert craters into appropriate sets
    CraterCreation.crater_do(FOVSize[0], SavePath, client)
    print('craters inserted \n')

    ### Normalize the images
    make_norm.norm_do(SavePath, norm_type = "both")
    print('normed \n')

    print('!!!!!!!!!!!!!!!!!!!')
    print(f'Time to run: {datetime.now() - t0}')
    print('all done')
    print('!!!!!!!!!!!!!!!!!!!')
