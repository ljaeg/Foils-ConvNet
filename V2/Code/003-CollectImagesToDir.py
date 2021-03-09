from PIL import Image  
import numpy as np 
import urllib.request
import requests
from io import BytesIO
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--AmazonFileName', default=os.path.join('..', 'Data', 'amazon20k.txt'))
parser.add_argument('--SaveDirName', default=os.path.join('..', 'Data', 'Predict', 'amazon20k'))
parser.add_argument('--MaxImages', default=None)
parser.add_argument('--Shuffle', default=False)
args = parser.parse_args()

# Make the output directory for saving images to disk.
if not os.path.exists(args.SaveDirName):
    os.mkdir(args.SaveDirName)

FileNames = np.genfromtxt(args.AmazonFileName, skip_header=1, dtype=str)

# Shuffling can be used to have multiple of these processes downloading at once.
if bool(args.Shuffle) == True:
    from random import shuffle
    shuffle(FileNames)

# MaxImages is used to do small runs if you only need some images.
if args.MaxImages is None:
    MaxImages = len(FileNames)
else:
    MaxImages = int(args.MaxImages)

print(f'Downloading {MaxImages} files from Amazon.')
for FileName in FileNames[:MaxImages]:
    OutputFileName = os.path.join(args.SaveDirName, FileName+'.jpg')
    # Check if the file exists.  If so, skip it.
    if os.path.exists(OutputFileName):
        # It has to exist AND not be zero bytes long.
        if os.stat(OutputFileName).st_size != 0:
            print('x', end='', flush=True)
            continue
    # time.sleep(0.5)
    print('.', end='', flush=True)
    url = "https://s3.amazonaws.com/stardustathome.testbucket/real/{x}/{x}-001.jpg".format(x=FileName)
    r = requests.get(url)
    with open(OutputFileName, 'wb') as f:
        f.write(r.content)
    # cmd = f"aws s3 cp s3://stardustathome.testbucket/real/{FileName}/{FileName}-001.jpg {OutputFileName}"
    # print(cmd)
    # os.system(cmd)
print()
print('Done!')
