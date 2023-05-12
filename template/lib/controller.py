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

        image_embedding = self.model.image_encoder(image_tensor)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = None,
            boxes  = None,
            masks  = None,
            )
        pred_masks, pred_ious = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            )

        image_size = [self.model.image_encoder.img_size]*2

        pred_masks = torch.sigmoid(pred_masks)
        ng_masks = self.model.postprocess_masks(pred_masks, image_size, original_size)

        ng_heatmap = np.squeeze(ng_masks.detach().cpu().numpy())
        bboxes = mt.get_xyxys_from_mask(ng_heatmap > .5)

        results = {
            'ng_heatmap': ng_heatmap,
            'bboxes': bboxes,
        }

        return results

    __call__ = infer