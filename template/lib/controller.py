import torch.nn.functional as F
from .segment_anything.custom.datasets import *
from .segment_anything import sam_model_registry
from .config import cfg
import pathlib


class Controller(object):
    def __init__(self) -> None:
        # initialize model
        self.device = 'cuda:{}'.format(mt.get_single_gpu_id())
        # Set up model
        curr_dir = pathlib.Path(__file__).parent.absolute()
        model_path = str(curr_dir / '../latest.pth')
        self.model = sam_model_registry[cfg.model_type](
            image_size=cfg.image_size, checkpoint=model_path, val=True).to(self.device)
        # setup image preprocessor
        self.preprocessor = Preprocessor(cfg.image_size, self.device)

    def infer(self, image):
        # preprocessing
        original_size = image.shape[:2]
        image_tensor = self.preprocessor(image)[None,...]

        # prediction
        image_embedding = self.model.image_encoder(image_tensor)
        logits = self.model.mask_decoder(
            image_embeddings=image_embedding,
            original_size=original_size
        )
        preds = F.softmax(logits, dim=1)
        preds = preds.detach().cpu().numpy()[0]
        ng_mask = 1. - preds[0,:,:]
        id_mask = np.argmax(preds, axis=0)

        bboxes = mt.get_xyxys_from_mask(id_mask)

        results = {
            'ng_mask': ng_mask,
            'seg_map': id_mask,
            'bboxes': bboxes,
        }

        return results

    __call__ = infer