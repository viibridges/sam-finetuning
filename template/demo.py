from lib.controller import Controller
import mtutils as mt

seg = Controller()

img = mt.cv_rgb_imread('test.jpg')
res = seg(img)

img = mt.draw_boxes(img, res['bboxes'], thickness=2, color=(255, 0, 0))
mt.PIS(img, res['ng_mask'], share_xy=True)