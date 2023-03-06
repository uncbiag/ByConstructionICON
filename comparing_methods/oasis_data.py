import random
import glob

data_path = "/playpen-ssd/lin.tian/data_local/oasis"

def get_data_list():
    return [total_img_list[i] for i in fixed_list], [total_segs_list[i] for i in fixed_list], [total_img_list[i] for i in moving_list], [total_segs_list[i] for i in moving_list]

def extract_id(img_path):
    return img_path.split('/')[-2].split('_')[-2]

# Test set starts from 0290, resulting 153 cases.
total_img_list = sorted(glob.glob(data_path + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))
total_segs_list = sorted(glob.glob(data_path + '/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz'))

total_img_list = list(filter(lambda x: extract_id(x)>='0290', total_img_list))
total_segs_list = list(filter(lambda x: extract_id(x)>='0290', total_segs_list))

assert len(total_img_list) == len(total_segs_list)

test_id_list = list(range(0, len(total_img_list)))

random.seed(1)
random.shuffle(test_id_list)
fixed_list = test_id_list[:5]
moving_list = test_id_list[5:]