import mtutils as mt

class Config(object):
    image_data_root = '/workspace/dataSet/dataset/sam-finetuning/'
    json_train = mt.osp.join(image_data_root, 'train.json')
    json_val   = mt.osp.join(image_data_root, 'val.json')
    json_test  = mt.osp.join(image_data_root, 'test.json')

    model_type = 'vit_b'  # vit_b, vit_l, vit_h, ascend in size
    checkpoint = mt.osp.join(image_data_root, 'checkpoints/sam_vit_b_01ec64.pth')

    image_size = 512
    batch_size = 64
    num_epochs = 100

    work_dir = 'tmp'
    model_path = mt.osp.join(work_dir, 'model_latest.pth')

cfg = Config()