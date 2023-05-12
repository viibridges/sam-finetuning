import mtutils as mt

class ConfigBaseModel(object):
    image_data_root = '/workspace/dataSet/dataset/sam-finetuning/'
    json_train = mt.osp.join(image_data_root, 'train.json')
    json_val   = mt.osp.join(image_data_root, 'val.json')
    json_test  = mt.osp.join(image_data_root, 'test.json')

    model_type = 'vit_b'  # vit_b, vit_l, vit_h, ascend in size
    checkpoint = mt.osp.join(image_data_root, 'checkpoints/sam_vit_b_01ec64.pth')

    image_size = 512
    batch_size = 64
    num_epochs = 100

    work_dir = 'tmp/base/'
    model_path = mt.osp.join(work_dir, 'latest.pth')


class ConfigLargeSize(ConfigBaseModel):
    image_size = 1024
    batch_size = 6
    work_dir = 'tmp/large/'
    model_path = mt.osp.join(work_dir, 'latest.pth')


class ConfigDebug(ConfigBaseModel):
    image_data_root = '/workspace/dataSet/dataset/sam-finetuning/'
    json_train = mt.osp.join(image_data_root, 'val.json')
    json_test  = mt.osp.join(image_data_root, 'val.json')
    batch_size = 2
    work_dir = '/tmp/debug/'
    model_path = mt.osp.join(work_dir, 'latest.pth')


cfg = ConfigLargeSize()