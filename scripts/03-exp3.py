import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
from constants import *
from models.model_bce import ModelBCE
import argparse


def test(path_to_images, model_to_test=None):
    output_dir_path = os.path.join(path_to_images, 'sal')
    list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*.jpg'))]
    print "predict shot %d frames" % (len(list_img_files))

    # Load Data
    list_img_files.sort()

    for curr_file in tqdm(list_img_files, ncols=20):
        curr_img_path = os.path.join(path_to_images, curr_file + '.jpg')
        print curr_img_path

        img = cv2.cvtColor(cv2.imread(curr_img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        predict(model=model_to_test, image_stimuli=img, name=curr_file, path_output_maps=output_dir_path)


def main(input_dir):
    # Create network
    model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    
    print ""
    print "Load weights..."
    # Here need to specify the epoch of model sanpshot
    load_weights(model.net['output'], path='gen_', epochtoload=90)

    print "Start predicting..."
    # Here need to specify the path to images and output path
    test(path_to_images=input_dir, model_to_test=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdir', help='Input directory', required=True)

    args = parser.parse_args()
    main(args.inputdir)
