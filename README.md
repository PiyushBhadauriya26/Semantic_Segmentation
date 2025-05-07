# Setup
Use Python 3.9
Use virtual env
Use requirements.txt to install the required python libs


# Server
- Start the server `python LitServe_SAM.py`
- Health check `http://localhost:8000/health`

# Client 
Use client code to call /predict API when the server is up.

```
usage: client.py [-h] --image IMAGE [--p1 P1] [--p2 P2] [--model MODEL] [--alpha ALPHA]

Send text & image to server and receive a response.

optional arguments:  
  -h, --help     show this help message and exit  
  --image IMAGE  URL for the image file.  
  --p1 P1        Single Point input in '(x,y)' format.
  --p2 P2        Point2 '(x1,y1)' for box input.
  --model MODEL  Model used for inference
  --alpha ALPHA  Transparency mask between 0-1.
```
response contains segmented image with identified region with no mask and green mask for background region.
### Example
- `python client.py --image .\data\test1.png --p1 "(60,40)" --p2 "(180,120)" --alpha 0.8`# For best results provide box input with Region of interest.
- `python client.py --image .\data\test1.png --alpha 0.5 --model "med_sam-vit_b"` # Use med_sam-vit_b model to segmentation whole image  
- `python client.py --image .\data\test1.png --p1 "(130,80)" --alpha 0.5 --model "med_sam-vit_b"` # Point input 