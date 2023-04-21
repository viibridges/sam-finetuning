from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from segment_anything.dataset.dataset import data_list_prepare, DataProvider
from segment_anything.modeling.common import LayerNorm2d
from segment_anything.loss import DiceLoss, FocalLoss
from mmcv.utils import get_logger
from functools import partial

import torch, argparse, os, mmcv, time
import torch.nn as nn
import mtutils as mt



def bulid_sam(
    encoder_embed_dim=1280,
    encoder_depth=32,
    encoder_num_heads=16,
    encoder_global_attn_indexes=[7, 15, 23, 31],
    checkpoint=None,
    if_finetune = False,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    if if_finetune:
        with torch.no_grad():
            image_encoder = ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
            )
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            )
    else:
        image_encoder = ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
    
    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


def parse_args(settings):
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--epoch', type=int, default=50, help='training epoches')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--device', help='the device to train models')
    parser.add_argument('--gups', type=int, default=None, help='the device ids to train models')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default=None, help='finetune checkpoint')
    parser.add_argument('--train_json', type=str, default=None, help='dataset train cnnjson path')
    parser.add_argument('--val_json', type=str, default=None, help='dataset val cnnjson path')
    parser.add_argument('--data_root', type=str, default=None, help='dataset data root path')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    args = parser.parse_args(settings)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def init_weights(model):
    # kaiming高斯初始化，使得每一卷积层的输出的方差都为1
    # torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
    # a ：Relu函数的负半轴斜率
    # mode：表示让前向传播还是反向传播的输出方差为1
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data,
                                    mode='fan_out',
                                    nonlinearity='relu')
        elif isinstance(m, LayerNorm2d):
            # Batchnorm层一共有两个需要学习的参数：
                # scale因子，初始化为1，shift因子，初始化为0
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def build_datasets(train_json, val_json, data_root, batch_size):
    train_data_list = data_list_prepare(train_json, data_root)
    val_data_list = data_list_prepare(val_json, data_root)
    train_dataloader, num_train_data = DataProvider.train_data_provide(train_data_list, batch_size)
    val_dataloader, num_val_data = DataProvider.val_data_provide(val_data_list, batch_size)
    return train_dataloader, val_dataloader, num_val_data


def batch_mask_iou(mask1, mask2):
    ious = list()
    for id, m1 in enumerate(mask1):
        mask_1 = m1[0]
        mask_2 = mask2[id][0]
        intersection = torch.matmul(mask_1, mask_2.t())
        area1 = torch.sum(mask_1, dim=1).view(1, -1)
        area2 = torch.sum(mask_2, dim=1).view(1, -1)
        union = (area1.t() + area2) - intersection
        iou = intersection / (union + 1e-6)
        sum_iou = iou.sum()
        ious.append(sum_iou.unsqueeze(0))
    mask_iou = torch.cat(ious)
    return mask_iou


class Loss_Function(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.iou_loss = torch.nn.MSELoss()
    
    def forward(self, pred, target, predict_ious):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        msk_iou = batch_mask_iou(pred, target)
        iou = self.iou_loss(predict_ious, msk_iou)
        loss = focal + .05 * dice + iou
        return loss


def train(settings):
    import multiprocessing as mp
    import torch.multiprocessing as t_mp

    mp.set_start_method('fork', force = True)
    start_method = mp.get_start_method()
    print(f"mp: {start_method}")

    t_mp.set_start_method('fork', force = True)
    start_method = t_mp.get_start_method()
    print(f"t_mp: {start_method}")

    args = parse_args(settings)
    torch.backends.cudnn.benchmark = True

    # create work_dir
    mmcv.mkdir_or_exist(mt.osp.abspath(args.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = mt.osp.join(args.work_dir, f'{timestamp}.log')
    logger = get_logger(name='segment anything', log_file=log_file, log_level='INFO')

    train_loader, val_loader, val_num = build_datasets(
        args.train_json,
        args.val_json,
        args.data_root,
        args.batch_size,
    )

    model = bulid_sam(checkpoint=args.checkpoint, if_finetune=True)
    init_weights(model)
    model.to(args.device)
    logger.info(model)

    loss_function = Loss_Function()

    iou_thres = 0.5

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e4, betas=(0.9, 0.999), weight_decay=0.1)

    best_accuracy = .0
    for epoch in mt.tqdm(range(args.epoch)):
        if_save_model = False

        model.train()
        running_loss = .0
        for step, data in enumerate(train_loader, start=0):
            images, masks, boxes = data
            input_data = list()
            for id, image in enumerate(images):
                input_data.append(dict(
                    image=image.to(args.device),
                    original_size=(600,600),
                    # point_coords=None,
                    # point_labels=None,
                    boxes=boxes[id].to(args.device),
                    mask_inputs=torch.unsqueeze(masks[id], dim=0).to(args.device)
                ))

            optimizer.zero_grad()

            predicts = model(input_data, False)
            batch_mask = list()
            batch_iou = list()
            for predict in predicts:
                batch_mask.append(predict["low_res_logits"])
                batch_iou.append(predict["iou_predictions"])
            predict_masks = torch.cat(batch_mask, 0)
            predict_ious = torch.cat(batch_iou, 0)
            loss = loss_function(predict_masks.to(dtype=torch.float32), masks.to(args.device), predict_ious)
            loss = loss.requires_grad_(True)
            loss.backward()

            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step+1)/len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
        print()

        # validate
        model.eval()
        acc = 0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in val_loader:
                images, masks, boxes = val_data
                input_data = list()
                for id, image in enumerate(images):
                    input_data.append(dict(
                        image=image.to(args.device),
                        original_size=(600,600),
                        # point_coords=None,
                        # point_labels=None,
                        boxes=boxes[id].to(args.device),
                        mask_inputs=torch.unsqueeze(masks[id], dim=0).to(args.device)
                    ))
                predict = model(input_data, False)
                for id, predict in enumerate(predicts):
                    predict_mask = predict["low_res_logits"]
                    iou = batch_mask_iou(predict_mask.to(dtype=torch.float32), masks[id].unsqueeze(0).to(args.device))
                    if iou >= iou_thres:
                        acc += 1

            val_accurate = acc / float(val_num)
            print(val_accurate)
            
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                (epoch + 1, running_loss / step, val_accurate))

        if val_accurate >= best_accuracy:
            if_save_model = True
        
        if if_save_model:
            torch.save(model.state_dict(), mt.osp.join(args.work_dir, 'latest.pth'))
    print('Finished Training')


if __name__ == "__main__":
    gpu_id, gpu_list = mt.get_gpu_str_as_you_wish(1, verbose=True)
    training_setting = [
        '--epoch', '50',
        '--work_dir', 'tmp/',
        '--device', 'cuda:1',
        '--gups', gpu_id,
        '--checkpoint', 'assets/sam-checkpoints/sam_vit_h_4b8939.pth',
        '--train_json', 'assets/train.json',
        '--val_json', 'assets/val.json',
        '--batch_size', '4',
        '--data_root', 'assets/sam标注数据/'
    ]
    train(training_setting)