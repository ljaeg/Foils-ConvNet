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

Dir = os.path.join("..", "Data") #directory where txt file is stored

# File name for full size images without craters
# fname = "amazon2k_nocraters.txt" #the txt file to get the code names from.
# save_f_base = "amazon2k_nocraters" #the hdf5 datafiles will be saved as SAVE_F_BASE_i where i goes from 0 to STEPS.
# fname = "amazon20k.txt" #the txt file to get the code names from.
# save_f_base = "amazon2k_no" #the hdf5 datafiles will be saved as SAVE_F_BASE_i where i goes from 0 to STEPS.

Save_to_Dir = Dir #where to save the hdf5 file to
# step_size = 2000 #How many images in a single datafile
steps = 1 #Number of hdf5 datafiles to make
# start_number = 2000 #In case you stopped a process earlier and want to pick up from where you left off.


def get_image(code):
        time.sleep(0.2) # If we don't throttle our reads, then Amazon will kick us out mid-download.
        url = "http://s3.amazonaws.com/stardustathome.testbucket/real/{x}/{x}-001.jpg".format(x=code)
        r = requests.get(url)
        try:
                img = Image.open(BytesIO(r.content))
                img = np.array(img) / 255.0
        except OSError:
                print("got error from URL: {}".format(url))
                img = np.ones((384, 512))
        try:
                # Sometimes the images come with channels, sometimes not.  Ensure homogeneity for the code below (i.e. we have channels -- then we choose the last channel).
                if len(img.shape) == 2:
                    img = img[..., np.newaxis]
                x = img[:384,:512,-1]
                assert x.shape == (384, 512), "error"
                print(".", flush=True, end='')
                return x
        except ValueError:
                print("failed reshape!")
                print(img.shape)
                print(url)
                return np.ones((384, 512))

#given a text file FNAME, start reading at line START and stop reading at START + STEP_SIZE, and create a big array of size (STEP_SIZE, 384, 512, 1) of the images.
def get_img_array(fname, start, step_size):
        path = os.path.join(Dir, fname)
        ims = []
        codes = []
        with open(path) as f:
                for line in f.read().splitlines()[start:start+step_size]:
                        code = str(line)
                        codes.append(code)
                        im = get_image(code)
                        ims.append(im)
        ims = np.array(ims, dtype = "f8")
        return ims, codes

#Create a single datafile with a single dataset (images) and a single attribute (codes)
def make_dataset(dataset_name, save_dir, codes_fname, start, step_size):
        ims, codes = get_img_array(codes_fname, start, step_size)
        datafile = h5py.File(os.path.join(save_dir, dataset_name + ".hdf5"), "w")
        image_set = datafile.create_dataset("images", ims.shape, data = ims, chunks=True, compression='gzip', compression_opts=2)
        datafile.create_dataset("amazoncodes", dtype=h5py.string_dtype(encoding='ascii'), data=[s.encode('ascii') for s in codes], compression='gzip', compression_opts=2)
        datafile.close()

#do_incrementally makes NUMBER_OF_STEPS number of different hdf5 files, so not all progress is lost if the code exits for some unforseen reason.
def do_incrementally(step_size, start, number_of_steps, save_dir, save_f_base, codes_fname):
        print("you are doing a total of {} images in {} parts".format(step_size*number_of_steps, number_of_steps))
        print("I predict a total time of something like {} minutes".format(predict_total_t(step_size * number_of_steps)))
        s = start 
        step = 0
        t = time.time()
        while step < number_of_steps:
                ds_name = save_f_base + "_" + str(step)
                make_dataset(ds_name, save_dir, codes_fname, s, step_size)
                print("saved " + ds_name)
                s += step_size
                step += 1
                time_til_done(step_size*number_of_steps, step*step_size, time.time() - t)

#rough esitimate of total time in minutes, ran at beginning 
def predict_total_t(total):
        return (total * 11.5) / (50 * 60)

#Ran incrementally to get how much time is left 
def time_til_done(total_N, current_N, current_time):
        total_t = (total_N * current_time) / current_N
        remaining_t = total_t - current_time
        print("seconds left: ", remaining_t)
        print("minutes left: ", remaining_t / 60)
        print("hours left: ", remaining_t / (60*60))
        print(" ")

#make a single dataset with train, test, val split. Not used in do_incrementally.
def make_train_test_val(split, save_file, txt_file, dataset_name):
        a = split[0]
        b = split[1]
        c = split[2]
        train, codes = get_img_array(txt_file, 0, a)
        print("got train")
        test, codes = get_img_array(txt_file, a, b)
        print("got test")
        val, codes = get_img_array(txt_file, a + b, c)
        print("got val")
        datafile = h5py.File(save_file + dataset_name + ".hdf5", "w")
        datafile.create_dataset("train", train.shape, data = train, chunks=True, compression='gzip', compression_opts=2)
        datafile.create_dataset("test", test.shape, data = test, chunks=True, compression='gzip', compression_opts=2)
        datafile.create_dataset("val", val.shape, data = val, chunks=True, compression='gzip', compression_opts=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txtfile', type=str, default='amazon20k.txt')
    parser.add_argument('--hdffile', type=str, default='amazon2k_no')
    parser.add_argument('--step_size', type=int, default='2000')
    parser.add_argument('--start_number', type=int, default='0')
    args = parser.parse_args()

    make_dataset(args.hdffile, Dir, args.txtfile, args.start_number, args.step_size)

    # Do incrementally if we are collecting a lot of images for analysis -- not training.
    # do_incrementally(step_size, start_number, steps, Dir, save_f_base, fname)

    # Old alternate method would download the files and split them.  This is now in a separate file so we can reuse this code regardless of whether we are splitting or not.
    # make_train_test_val([2500, 500, 500], Dir, fname, "NO_TRAIN")

    print("DONE!")





















