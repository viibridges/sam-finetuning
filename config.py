import mtutils as mt

class ConfigBaseModel(object):
    image_data_root = '/workspace/dataSet/dataset/sam-finetuning/'
    json_train = mt.osp.join(image_data_root, 'train.json')
    json_val   = mt.osp.join(image_data_root, 'val.json')
    json_test  = mt.osp.join(image_data_root, 'test.json')

    model_type = 'vit_b'  # vit_b, vit_l, vit_h, ascend in size
    checkpoint = mt.osp.join(image_data_root, 'checkpoints/sam_vit_b_0b3195.pth')

    image_size = 512
    batch_size = 64
    num_epochs = 100

    work_dir = 'tmp/base/'
    model_path = mt.osp.join(work_dir, 'latest.pth')


class ConfigLargeModel(ConfigBaseModel):
    image_data_root = '/workspace/dataSet/dataset/sam-finetuning/'
    model_type = 'vit_l'  # vit_b, vit_l, vit_h, ascend in size
    checkpoint = mt.osp.join(image_data_root, 'checkpoints/sam_vit_l_0b3195.pth')
    batch_size = 4
    work_dir = 'tmp/large/'
    model_path = mt.osp.join(work_dir, 'latest.pth')


cfg = ConfigLargeModel()