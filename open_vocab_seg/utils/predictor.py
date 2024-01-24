# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import numpy as np
import torch
from torch.nn import functional as F
import cv2

from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.modeling.postprocessing import sem_seg_postprocess

import open_clip
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry 
from open_vocab_seg.modeling.clip_adapter.adapter import PIXEL_MEAN, PIXEL_STD
from open_vocab_seg.modeling.clip_adapter.utils import crop_with_mask

class OVSegPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, original_image, class_names):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width, "class_names": class_names}
            predictions = self.model([inputs])[0]
            return predictions

class OVSegVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, class_names=None):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.class_names = class_names

    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        class_names = self.class_names if self.class_names is not None else self.metadata.stuff_classes

        for label in filter(lambda l: l < len(class_names), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (sem_seg == label).astype(np.uint8)
            text = class_names[label]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=(1.0, 1.0, 240.0 / 255),
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output



class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            raise NotImplementedError
        else:
            self.predictor = OVSegPredictor(cfg)

    def run_on_image(self, image, class_names):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        predictions = self.predictor(image, class_names)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        if "sem_seg" in predictions:
            r = predictions["sem_seg"]
            blank_area = (r[0] == 0)
            pred_mask = r.argmax(dim=0).to('cpu')
            pred_mask[blank_area] = 255
            pred_mask = np.array(pred_mask, dtype=np.int)

            vis_output = visualizer.draw_sem_seg(
                pred_mask
            )
        else:
            raise NotImplementedError

        return predictions, vis_output
    
class SAMVisualizationDemo(object):
    def __init__(self, cfg, granularity, sam_path, ovsegclip_path, instance_mode=ColorMode.IMAGE, parallel=False):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        self.granularity = granularity
        sam = sam_model_registry["vit_l"](checkpoint=sam_path).cuda()
        self.predictor = SamAutomaticMaskGenerator(sam, points_per_batch=16)
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=ovsegclip_path)

    def run_on_image(self, ori_image, class_names):
        height, width, _ = ori_image.shape
        if width > height:
            new_width = 1280
            new_height = int((new_width / width) * height)
        else:
            new_height = 1280
            new_width = int((new_height / height) * width)
        image = cv2.resize(ori_image, (new_width, new_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        visualizer = OVSegVisualizer(ori_image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        with torch.no_grad(), torch.cuda.amp.autocast():
            masks = self.predictor.generate(image)
        pred_masks = [masks[i]['segmentation'][None,:,:] for i in range(len(masks))]
        pred_masks = np.row_stack(pred_masks)
        pred_masks = BitMasks(pred_masks)
        bboxes = pred_masks.get_bounding_boxes()

        mask_fill = [255.0 * c for c in PIXEL_MEAN]

        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        regions = []
        for bbox, mask in zip(bboxes, pred_masks):
            region, _ = crop_with_mask(
                image,
                mask,
                bbox,
                fill=mask_fill,
            )
            regions.append(region.unsqueeze(0))
        regions = [F.interpolate(r.to(torch.float), size=(224, 224), mode="bicubic") for r in regions]

        pixel_mean = torch.tensor(PIXEL_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(PIXEL_STD).reshape(1, -1, 1, 1)
        imgs = [(r/255.0 - pixel_mean) / pixel_std for r in regions]
        imgs = torch.cat(imgs)
        if len(class_names) == 1:
            class_names.append('others')
        txts = [f'a photo of {cls_name}' for cls_name in class_names]
        text = open_clip.tokenize(txts)

        img_batches = torch.split(imgs, 32, dim=0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            self.clip_model.cuda()
            text_features = self.clip_model.encode_text(text.cuda())
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features = []
            for img_batch in img_batches:
                image_feat = self.clip_model.encode_image(img_batch.cuda().half())
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_features.append(image_feat.detach())
            image_features = torch.cat(image_features, dim=0)
            class_preds = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        select_cls = torch.zeros_like(class_preds)

        max_scores, select_mask = torch.max(class_preds, dim=0)
        if len(class_names) == 2 and class_names[-1] == 'others':
            select_mask = select_mask[:-1]
        if self.granularity < 1:
            thr_scores = max_scores * self.granularity
            select_mask = []
            if len(class_names) == 2 and class_names[-1] == 'others':
                thr_scores = thr_scores[:-1]
            for i, thr in enumerate(thr_scores):
                cls_pred = class_preds[:,i]
                locs = torch.where(cls_pred > thr)
                select_mask.extend(locs[0].tolist())
        for idx in select_mask:
            select_cls[idx] = class_preds[idx]
        semseg = torch.einsum("qc,qhw->chw", select_cls.float(), pred_masks.tensor.float().cuda())

        r = semseg
        blank_area = (r[0] == 0)
        pred_mask = r.argmax(dim=0).to('cpu')
        pred_mask[blank_area] = 255
        pred_mask = np.array(pred_mask, dtype=np.int)
        pred_mask = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        vis_output = visualizer.draw_sem_seg(
            pred_mask
        )

        return None, vis_output