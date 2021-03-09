#This is a program for going thru the txt file and collecting images of craters 
#To store on the machine at Berkeley.
#Images are saved in an hdf5 file, with the codes stored in an attribute of the hdf5 file.

from PIL import Image  
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import time
import os
import argparse
import glob2 as glob

def GetImageFromFile(FileName):
    # First load the file from disk.
    try:
        img = Image.open(FileName)
        img = np.array(img) / 255.0
    except OSError:
        img = np.ones((384, 512))
    # Ensure the image has the right dimensions for us.
    try:
        # Sometimes the images come with channels, sometimes not.  Ensure homogeneity for the code below (i.e. we have channels -- then we choose the last channel).
        if len(img.shape) == 2:
            img = img[..., np.newaxis]
        x = img[:384,:512,-1]
        assert x.shape == (384, 512), "error"
        return x
    except ValueError:
        print("failed reshape!")
        print(img.shape)
        return np.ones((384, 512))


def HDFFromDir(DirName, HDFName):
    FileNames = glob.glob(os.path.join(DirName, '*.jpg'))
    AmazonCodes = [os.path.splitext(os.path.basename(FileName))[0] for FileName in FileNames]

    print('\nWriting them to HDF.')

    # Create the HDF file.
    ImageArrayShape = (len(AmazonCodes), 384, 512)
    DataFile = h5py.File(HDFName + ".hdf5", "w")
    image_set = DataFile.create_dataset("images", ImageArrayShape, chunks=(1,384,512), compression='gzip', compression_opts=2)
    amazoncodes = DataFile.create_dataset("amazoncodes", (len(AmazonCodes),), dtype=h5py.string_dtype(encoding='ascii'), compression='gzip', compression_opts=2)

    # Loop through all the files -- read them in and insert them into the HDF.
    for i,FileName in enumerate(FileNames):
        print(".", flush=True, end='')
        Image = GetImageFromFile(FileName)
        image_set[i,...] = np.array(Image, dtype='f8')
        amazoncodes[i] = [AmazonCodes[i].encode('ascii'),]
        # DataFile.flush()

    DataFile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ImagesDir', type=str, default=os.path.join('..', 'Data', 'amazon10k_nocraters'))
    parser.add_argument('--HDFName', type=str, default='amazon10k_FullFOV_nocraters')
    args = parser.parse_args()

    print("START!")

    HDFFromDir(args.ImagesDir, args.HDFName)

    print("\nDONE!")
