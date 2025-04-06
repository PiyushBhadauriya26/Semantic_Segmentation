import litserve as ls
import torch
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything import SamPredictor, sam_model_registry
from io import BytesIO
from PIL import Image
from fastapi import Response
import numpy as np

# device = torch.device(
#     'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
# )
class Same2API(ls.LitAPI):
    def setup(self, device):
        # Load the model
        # self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        model_checkpoint = "model_checkpoint/sam_vit_l_0b3195.pth"
        sam = sam_model_registry["vit_l"](checkpoint=model_checkpoint)
        self.predictor = SamPredictor(sam)


    def decode_request(self, request):
        try :
            point1 = tuple(map(int, request["p1"].strip('()').split(','))) if "p1" in request else None
            point2 = tuple(map(int, request["p2"].strip('()').split(','))) if "p2" in request else None
        except:
            return "Invalid input format. Please provide point p1 and p2 in format '(x,y)'"
        alpha = float(request["alpha"]) if "alpha" in request else None
        return (request["image"].file.read(), point1, point2, alpha)
    def predict(self, input):
        original_image = Image.open(input[0])
        point1 = input[1]
        point2 = input[2]
        print(point1)
        print(point2)
        original_image.show()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(np.array(original_image.convert("RGB")))
            if point1 and point2: # box input
                masks, _, _ = self.predictor.predict(
                    box=np.array([point1[0], point1[1],point2[0], point2[1]]),
                    multimask_output=False,
                    )
            elif point1 and not point2: # point input
                masks, _, _ = self.predictor.predict(
                    point_coords=np.array([[point1[0], point1[1]]]), # x, y coordinate of the point
                    point_labels=np.array([1]), # 1 = foreground
                    multimask_output=False,
                    )
            else: # no input Auto generate mask using whole image as box
                masks, _, _ = self.predictor.predict(
                    box=np.array([0, 0, original_image.size[0], original_image.size[1]]),
                    multimask_output=False,
                    )

            # Extract the mask
            alpha = input[3] if (0 <= input[3] <= 1) else 0.8
            tp = int(225 * alpha)
            mask_image = Image.fromarray(((masks[0] * (225-tp)) + tp).astype(np.uint8))
            # Paste the original image into the cutout image, using the mask as the alpha channel
            cutout_image = Image.new('RGBA', original_image.size, color="green")
            cutout_image.paste(original_image, (0, 0), mask=mask_image)
            cutout_image.show()
            return cutout_image

    def encode_response(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return Response(content=buffered.getvalue(), headers={"Content-Type": "image/png"})


# Starting the server
if __name__ == "__main__":
    api = Same2API()
    server = ls.LitServer(api, timeout=False, track_requests=True)
    server.run(port=8000)