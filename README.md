#### Setup
- Use Python 3.12
- Use virtual env
- Install pytorch https://pytorch.org/get-started/locally/#windows-python
- Use requirements.txt to install the required python libs

#### Server
- Start the server `python LitServe_SAM.py`
- Health check `http://localhost:8000/health`

#### Client 
Use client code to call /predict API when the server is up.

```
usage: client.py [-h] --image IMAGE [--p1 P1] [--p2 P2] [--model MODEL] [--alpha ALPHA]

Send text & image to server and receive a response.

optional arguments:  
  -h, --help     show this help message and exit  
  --image IMAGE  Path or URL for the image file..  
  --p1 P1        Single Point input in '(x,y)' format.
  --p2 P2        Point2 '(x1,y1)' for box input.
  --model MODEL  Name of the model [sam-vit_l, sam-vit_h, med_sam-vit_b, sam-vit_b-lora512]
  --alpha ALPHA  Transparency mask between 0-1.
```
response contains segmented image with identified region with no mask and green mask for background region.
###### Example
- `python client.py --image data/test.png --p1 "(120,70)" --p2 "170,120" --alpha 0.5 --model "sam-vit_l" `# Provide box input with Region of interest.
- `python client.py --image data/test.png --alpha 0.5 --model "med_sam-vit_b"` # Use med_sam-vit_b model to segment whole image  
- `python client.py --image data/test.png --p1 "(145,110)" --alpha 0.5 --model sam-vit_b-lora512` # use Point input with FineTuned sam-vit_b-lora512 model 

#### Batch Inference 
Use `batch_inference.py` script to run inference on multiple images
```
usage: batch_inference.py [-h] [-i DATA_PATH] [-o SEG_PATH] [--device DEVICE] [--overwrite OVERWRITE] [--model MODEL]

run inference on testing set based on MedSAM

options:
  -h, --help                               show this help message and exit
  -i DATA_PATH, --data_path DATA_PATH      path to the data folder
  -o SEG_PATH, --seg_path SEG_PATH         path to the segmentation folder
  --device DEVICE                          Device cuda or cpu
  --overwrite OVERWRITE                    Overwrite existing results with a new mask.
  --model MODEL                            Name of the model [sam-vit_l, sam-vit_h, med_sam-vit_b, sam-vit_b-lora512]

```
###### Example
- `python .\batch_inference.py -i "data/BCSS_small/test/images" --overwrite True` # Run inference and save predicted masks in data/Results folder

#### Train
- Source: https://github.com/WangRongsheng/SAM-fine-tune
- Update `config.yaml` for DATASET paths, CHECKPOINT for base sam model and TRAIN setting
- `python train.py` # Start training.
- After training lora weights are saved as safetensors file in model_checkpoint folder

#### References
1. https://github.com/facebookresearch/segment-anything
2. https://github.com/facebookresearch/sam2
3. https://github.com/bowang-lab/MedSAM
4. https://github.com/mazurowski-lab/finetune-SAM
5. https://github.com/WangRongsheng/SAM-fine-tune
6. https://github.com/Lightning-AI/LitServe
