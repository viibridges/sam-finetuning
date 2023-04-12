from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import argparse

def parse_args():
    parse=argparse.ArgumentParser(description="mmse test the images or videos")
    parse.add_argument("--model_type",default='vit_h' ,help="vit_b, vit_l, vit_h, ascend in size")
    parse.add_argument("--checkpoint_path",default='model/sam_vit_h_4b8939.pth', help='model/sam_vit_b_01ec64.pth, model/sam_vit_l_0b3195.pth, model/sam_vit_h_4b8939.pth')
    parse.add_argument("--img_path",default='/home/dongxinyu/Documents/data_manage/96gaodu/meidong0410/imgs/Meidong_SeaLand_Hmeidong_96_20230323163423-_20230323163611_image_0001.jpg')
    parse.add_argument("--mode",default='everything', help='prompt, everything')

    args=parse.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('loading checkpoint')
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path,val=True)
    print('reading img')
    myimg = cv2.imread(args.img_path)

    # with prompt
    if args.mode == 'prompt':
        print('loading predictor')
        predictor = SamPredictor(sam)
        predictor.set_image(myimg)
        # box prompts
        box_prompts = np.array([0, 0, 1920, 1080])
        masks, _, _ = predictor.predict(box=box_prompts)
        
        # # point promps
        # point_coords_prompts = np.array([])
        # point_labels_prompts = np.array([])
        # a = np.linspace(0,1920,20)
        # b = np.linspace(0,1080,20)
        # for i in range(len(a)):
        #     point_coords_prompts = np.append(point_coords_prompts,np.array([a[i],b[i]]))
        #     point_labels_prompts = np.append(point_labels_prompts,np.random.randint(2,size=1))
        # point_coords_prompts = point_coords_prompts.reshape(-1,2)
        # print(point_coords_prompts)
        # masks, _, _ = predictor.predict(point_coords=point_coords_prompts, point_labels=point_labels_prompts)

        emptyimg = np.zeros(myimg.shape, np.uint8)
        # 浅灰色背景
        emptyimg.fill(200)

        # print(masks)
        for idx, mask in enumerate(masks):
            pallete = np.random.randint(256,size=(3),dtype=np.uint8)
            emptyimg[mask] = pallete
        combine = cv2.addWeighted(emptyimg, 0.5, myimg, 0.5, 0)
        cv2.imwrite('output/test.jpg',combine)


    # everything
    elif args.mode == 'everything':
        print('loading mask_generator')
        mask_generator = SamAutomaticMaskGenerator(sam)
        print('generating mask')
        masks = mask_generator.generate(myimg)

        emptyimg = np.zeros(myimg.shape, np.uint8)
        # 浅灰色背景
        emptyimg.fill(200)

        # print(masks)
        for idx, mask_dict in enumerate(masks):
            mask = mask_dict['segmentation']
            pallete = np.random.randint(256,size=(3),dtype=np.uint8)
            emptyimg[mask] = pallete
        combine = cv2.addWeighted(emptyimg, 0.5, myimg, 0.5, 0)
        cv2.imwrite('output/test.jpg',combine)