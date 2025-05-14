# -*- coding: utf-8 -*-

"""
usage example:
python batch_inference.py -i "data/BCSS_small/train/images" --model=sam-vit_l
"""

# %% load environment
import numpy as np
import os
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import argparse
from LitServe_SAM import models, lora_parms
from src.lora import LoRA_sam


@torch.no_grad()
def batch_inference(model_name, in_folder, out_folder, overwrite=False):
    model_checkpoint = models[model_name]
    sam = sam_model_registry[model.split("-")[1]](checkpoint=model_checkpoint)
    if len(model.split("-")) == 3 and model.split("-")[2].startswith("lora"):
        sam_lora = LoRA_sam(sam, 512)
        sam_lora.load_lora_parameters(lora_parms[model])
        sam = sam_lora.sam
    predictor = SamPredictor(sam)
    in_images = [file for file in os.listdir(in_folder) if file.endswith('.png')]
    for image in in_images:
        if os.path.isfile(os.path.join(out_folder, model, image)) and not overwrite:
            continue
        original_image = Image.open(os.path.join(in_folder, image))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(np.array(original_image.convert("RGB")))
            masks, pred_quality, _ = predictor.predict(
                box=np.array([0, 0, original_image.size[0], original_image.size[1]]),
                multimask_output=True
            )
            print(pred_quality)
        mask_image = Image.fromarray(((masks[0]).astype(np.uint8)))
        os.makedirs(os.path.join(out_folder, model), exist_ok=True)
        mask_image.save(os.path.join(out_folder, model, image))


parser = argparse.ArgumentParser(
    description="run inference on testing set based on MedSAM"
)
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="data/BCSS_small/images",
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="data/Results",
    help="path to the segmentation folder",
)

parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument("--overwrite", type=bool, default=False, help="Overwrite existing results with a new mask.")
parser.add_argument(
    "--model",
    type=str,
    default="med_sam-vit_b",
    help="Name of the model [sam-vit_l, sam-vit_h, med_sam-vit_b, sam-vit_b-lora512]"
)
args = parser.parse_args()
device = args.device
model = args.model
overwrite = args.overwrite
data_path = args.data_path
seg_path = args.seg_path

batch_inference(model, data_path, seg_path, overwrite)
