from shutil import copyfile
import os
from glob import glob

val_names = [os.path.basename(name).split('_')[0] for name in
             glob(r'P:\Captcha_data\Captcha.v1i.yolov8_objects\valid\images\*.jpg')]

val_names = list(set(val_names))

data_path = r'P:\mixed_train_to_the_coordinates_dataset\mixed_train_to_the_coordinates_dataset'

save_path = r'/val_dataset'
for name in val_names:
    img = os.path.join(data_path, name + '.jpg')
    markup = os.path.join(data_path, name + '.json')

    copyfile(os.path.join(img), os.path.join(save_path, os.path.basename(img)))
    copyfile(os.path.join(markup), os.path.join(save_path, os.path.basename(markup)))