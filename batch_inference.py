# -*- coding: utf-8 -*-

"""
usage example:
python batch_Inference_SS.py -i assets/img_demo.png -o ./ --model

"""

# %% load environment
import numpy as np
import os

join = os.path.join
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import argparse
from LitServe_SAM import models

@torch.no_grad()
def batch_inference(model_name, in_folder, out_folder):
    model_checkpoint = models[model_name]
    sam = sam_model_registry[model.split("-")[1]](checkpoint=model_checkpoint)
    predictor = SamPredictor(sam)
    in_images = [file for file in os.listdir(in_folder) if file.endswith('.png')]
    for image in in_images:
        original_image = Image.open(os.path.join(in_folder, image))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(np.array(original_image.convert("RGB")))
            masks, _, _ = predictor.predict(
                box=np.array([0, 0, original_image.size[0], original_image.size[1]]),
                multimask_output=False
            )
        mask_image = Image.fromarray(((masks[0] * 225).astype(np.uint8)))
        os.makedirs(os.path.join(out_folder, model), exist_ok=True)
        mask_image.save(os.path.join(out_folder, model, image))


parser = argparse.ArgumentParser(
    description="run inference on testing set based on MedSAM"
)
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="data\BCSS_small\images",
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="data\Results",
    help="path to the segmentation folder",
)

parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "--model",
    type=str,
    default="med_sam-vit_b",
    help="Name to the model",
)
args = parser.parse_args()
device = args.device
model = args.model
data_path = args.data_path
seg_path = args.seg_path

batch_inference(model, data_path, seg_path)
