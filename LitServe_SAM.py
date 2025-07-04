import litserve as ls
import torch
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything import SamPredictor, sam_model_registry
from src.lora import LoRA_sam
from io import BytesIO
from PIL import Image, ImageDraw
from fastapi import Response
import numpy as np
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor

# device = torch.device(
#     'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
# )
models = {"sam-vit_l": "model_checkpoint/sam_vit_l_0b3195.pth",
          "sam-vit_h": "model_checkpoint/sam_vit_h_4b8939.pth",
          "med_sam-vit_b": "model_checkpoint/MedSAM/medsam_vit_b.pth",
          "sam-vit_b-lora512": "model_checkpoint/sam_vit_b_01ec64.pth"}
lora_parms = {"sam-vit_b-lora512": "model_checkpoint/sam_vit_b-lora512.safetensors"}

def is_tuple_of_ints(var):
    """
    Checks if a variable is a tuple and contains only integers.

    Args:
        var: The variable to check.

    Returns:
        True if the variable is a tuple and contains only integers, False otherwise.
    """
    if not isinstance(var, tuple):
        return False
    for element in var:
        if not isinstance(element, int):
            return False
    return True


class Sam_API(ls.LitAPI):
    def __init__(self):
        self.predictors = {}

    def setup(self, device):
        # Load the model
        # self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        for model, model_checkpoint in models.items():
            # if model.startswith("sam2"):
            #     model_cfg = "model_checkpoint/sam2.1_hiera_s.yaml"
            #     predictor = SAM2ImagePredictor(build_sam2(model_cfg, model_checkpoint))
            # else:
            sam = sam_model_registry[model.split("-")[1]](checkpoint=model_checkpoint)
            if len(model.split("-")) == 3 and model.split("-")[2].startswith("lora"):
                sam_lora = LoRA_sam(sam, 512)
                sam_lora.load_lora_parameters(lora_parms[model])
                sam = sam_lora.sam
            predictor = SamPredictor(sam)
            self.predictors[model] = predictor

    def decode_request(self, request):
        print(request)
        try:
            # Allow the point parameters to be (x,y) or [x,y] as well as remove whitespaces, since bash doesn't escape ()s
            point1 = tuple(map(int, request["p1"].strip('()[] ').split(','))) if "p1" in request else None
            point2 = tuple(map(int, request["p2"].strip('()[] ').split(','))) if "p2" in request else None
        except:
            return "Invalid input format. Please provide point p1 and p2 in format '(x,y)'"
        alpha = float(request["alpha"]) if "alpha" in request else None
        model = request["model"] if "model" in request else "med_sam-vit_b"
        if point1 and not is_tuple_of_ints(point1):
            return "Invalid point input. p1={}".format(point1)
        elif (not point1 and point2) or (point2 and not is_tuple_of_ints(point2)):
            return "Invalid box input. p1={},p2={}".format(point1, point2)
        elif model not in self.predictors:
            return "Invalid Model input. {}, available models = {}".format(model, self.predictors)
        return (request["image"].file.read(), point1, point2, model, alpha)


    def predict(self, input):
        if len(input) != 5:
            return input
        original_image = Image.open(input[0])
        point1 = input[1]
        point2 = input[2]
        model = input[3]
        alpha = input[4] if (0 <= input[4] <= 1) else 0.8
        print(model)
        # print(point2)
        # original_image.show()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictors[model].set_image(np.array(original_image.convert("RGB")))
            if point1 and point2:  # box input
                masks, pred_quality, _ = self.predictors[model].predict(
                    box=np.array([point1[0], point1[1], point2[0], point2[1]]),
                    multimask_output=True,
                )
            elif point1 and not point2:  # point input
                masks, pred_quality, _ = self.predictors[model].predict(
                    point_coords=np.array([[point1[0], point1[1]]]),  # x, y coordinate of the point
                    point_labels=np.array([1]),  # 1 = foreground
                    multimask_output=True,
                )
            else:  # no input Auto generate mask using whole image as box
                masks, pred_quality, _ = self.predictors[model].predict(
                    box=np.array([0, 0, original_image.size[0], original_image.size[1]]),
                    multimask_output=True,
                )

            tp = int(255 * alpha)
            # Extract the mask with max pred_quality
            mask_image = Image.fromarray(((masks[np.argmax(pred_quality)] * (255 - tp)) + tp).astype(np.uint8))
            # Paste the original image into the cutout image, using the mask as the alpha channel
            cutout_image = Image.new('RGBA', original_image.size, color="green")
            cutout_image.paste(original_image, (0, 0), mask=mask_image)
            if point1 and point2:  # box input
                draw = ImageDraw.Draw(cutout_image)
                draw.rectangle((point1[0], point1[1], point2[0], point2[1]), outline=(0, 0, 0))
            if point1 and not point2:  # point input
                draw = ImageDraw.Draw(cutout_image)
                # draw.point((point1[0], point1[1]), fill=(0, 0, 0))
                draw.circle((point1[0], point1[1]),radius=2, fill=(0, 0, 0))
            cutout_image.show()
            return cutout_image

    def encode_response(self, content):
        if isinstance(content, Image.Image):
            buffered = BytesIO()
            content.save(buffered, format="PNG")
            return Response(content=buffered.getvalue(), headers={"Content-Type": "image/png"})
        elif isinstance(content, str):
            return Response(content=content, status_code=400, headers={"Content-Type": "text/plain"})
        else:
            print(content)
            return Response(content="Internal Server Error", status_code=500, headers={"Content-Type": "text/plain"})


# Starting the server
if __name__ == "__main__":
    api = Sam_API()
    server = ls.LitServer(api, timeout=False, track_requests=True, model_metadata=models)
    server.run(port=8000)
