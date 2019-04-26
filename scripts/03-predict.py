import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
from constants import *
from models.model_bce import ModelBCE
import argparse


def test(path_to_images, shot_counts, model_to_test=None):
    for shot_idx in range(1, int(shot_counts)+1):
        shot_dir = 'shot-%d' % shot_idx
        shot_img_dir = os.path.join(shot_dir, 'frame_images')
        shot_sal_dir = os.path.join(shot_dir, 'saliency_maps')

        list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, shot_img_dir, '*'))]

        print "predict %d-th shot from %s to %s: %d frames" % (shot_idx, shot_img_dir, shot_sal_dir, len(list_img_files))

        # Load Data
        list_img_files.sort()

        for curr_file in tqdm(list_img_files, ncols=20):
            curr_img_path = os.path.join(path_to_images, shot_img_dir, curr_file + '.jpg')
            print curr_img_path

            img = cv2.cvtColor(cv2.imread(curr_img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            predict(model=model_to_test, image_stimuli=img, name=curr_file, path_output_maps=shot_sal_dir)


def main(input_dir, shot_counts):
    # Create network
    model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    
    print ""
    print "Load weights..."
    # Here need to specify the epoch of model sanpshot
    load_weights(model.net['output'], path='gen_', epochtoload=90)

    print "Start predicting..."
    # Here need to specify the path to images and output path
    test(path_to_images=input_dir, shot_counts=shot_counts, model_to_test=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdir', help='Input directory', required=True)
    parser.add_argument('-c', '--shotcounts', help='Shot Counts', required=True)

    args = parser.parse_args()
    main(args.inputdir, args.shotcounts)
