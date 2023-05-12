#
# Release you model
#
import os
from config import cfg

target_dir = os.path.join(cfg.work_dir, 'release/')
os.makedirs(target_dir, exist_ok=True)

print("Publishing...")
os.system('cp -rL template/* {}'.format(target_dir))
os.system('cp {} {}/latest.pth'.format(cfg.model_path, target_dir))
print("Done! Your release folder locates in: {}".format(target_dir))