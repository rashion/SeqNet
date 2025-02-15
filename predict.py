import argparse
import os
import tensorflow as tf
from utils import define_model, crop_prediction, lowercase
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
import cv2
from PIL import Image

from tensorflow.keras import backend as K
from tensorflow.keras.layers import ReLU

tf.compat.v1.disable_v2_behavior()

def save_mask(pred_, output_path, filename, image_size, mask_type, crop_size, stride_size, new_height, new_width):
    pred_ = crop_prediction.recompone_overlap(pred_, crop_size, stride_size, new_height, new_width)
    pred_ = pred_[:, 0:576, 0:576, :]
    pred_ = pred_[0, :, :, 0]
    pred_ = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
    pred_ = resize(pred_, image_size[::-1])
    cv2.imwrite(f"{output_path}/{mask_type}/{filename}.png", pred_)
    del pred_

def save_final_output(output_path, filename, pred_seg_mask_path, pred_art_mask_path, pred_vei_mask_path):
    # Load masks
    pred_seg_mask = cv2.imread(pred_seg_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_art_mask = cv2.imread(pred_art_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_vei_mask = cv2.imread(pred_vei_mask_path, cv2.IMREAD_GRAYSCALE)

    pred_final = np.zeros((*pred_seg_mask.shape, 3), dtype=np.uint8)
    pred_final[pred_art_mask >= pred_vei_mask, 2] = pred_seg_mask[pred_art_mask >= pred_vei_mask]
    pred_final[pred_art_mask < pred_vei_mask, 0] = pred_seg_mask[pred_art_mask < pred_vei_mask]

    # Save final output
    cv2.imwrite(f"{output_path}/out_final/{filename}.png", pred_final)


def predict(ACTIVATION='ReLU', dropout=0.1, batch_size=32, repeat=4, minimum_kernel=32,
            epochs=200, iteration=3, crop_size=128, stride_size=3,
            input_path='', output_path='', DATASET='ALL'):
    exts = ['png', 'jpg', 'tif', 'bmp', 'gif']
    lowercase.convert_filenames_to_lowercase(input_path)

    if not input_path.endswith('/'):
        input_path += '/'
    paths = [input_path + i for i in sorted(os.listdir(input_path)) if i.split('.')[-1] in exts]

    os.makedirs(f"{output_path}/out_seg/", exist_ok=True)
    os.makedirs(f"{output_path}/out_art/", exist_ok=True)
    os.makedirs(f"{output_path}/out_vei/", exist_ok=True)
    os.makedirs(f"{output_path}/out_final/", exist_ok=True)

    activation = globals()[ACTIVATION]
    model = define_model.get_unet(minimum_kernel=minimum_kernel, do=dropout, activation=activation, iteration=iteration)
    model_name = f"Final_Emer_Iteration_{iteration}_cropsize_{crop_size}_epochs_{epochs}"
    print("Model : %s" % model_name)
    load_path = f"trained_model/{DATASET}/{model_name}.hdf5"
    model.load_weights(load_path, by_name=False)

    for i in tqdm(range(len(paths))):
        filename = '.'.join(paths[i].split('/')[-1].split('.')[:-1])
        img = Image.open(paths[i])
        image_size = img.size
        img = np.array(img) / 255.
        img = resize(img, [576, 576])
        patches_pred, new_height, new_width, _ = crop_prediction.get_test_patches(img, crop_size, stride_size)
        del img, _
        preds = model.predict(patches_pred)

        # Save masks
        save_mask(preds[iteration], output_path, filename, image_size, "out_seg", crop_size, stride_size, new_height, new_width)
        save_mask(preds[2 * iteration + 1], output_path, filename, image_size, "out_art", crop_size, stride_size, new_height, new_width)
        save_mask(preds[3 * iteration + 2], output_path, filename, image_size, "out_vei", crop_size, stride_size, new_height, new_width)

        # Calculate final output
        save_final_output(output_path, filename, f"{output_path}/out_seg/{filename}.png", f"{output_path}/out_art/{filename}.png", f"{output_path}/out_vei/{filename}.png")

if __name__ == "__main__":
    #  os.environ["CUDA_VISIBLE_DEVICES"]="0"
    #  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    des_text = 'Please use -i to specify the input dir and -o to specify the output dir.'
    parser = argparse.ArgumentParser(description=des_text)
    parser.add_argument('--input', '-i', help="(Required) Path of input dir")
    parser.add_argument('--output', '-o', help="(Optional) Path of output dir")
    args = parser.parse_args()

    if not args.input:
        print('Please specify the input dir with -i')
        exit(1)

    input_path = args.input
    output_path = args.output if args.output else './output'

    if output_path.endswith('/'):
        output_path = output_path[:-1]

    predict(batch_size=24, epochs=200, iteration=3, stride_size=3, crop_size=128,
            input_path=input_path, output_path=output_path)
