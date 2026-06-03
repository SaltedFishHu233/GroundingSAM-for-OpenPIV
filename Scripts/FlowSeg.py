import os
import cv2
import torch
import numpy as np
import sys

from groundingdino.util import box_ops
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict as dino_predict
from groundingdino.util.inference import annotate, load_image
from huggingface_hub import hf_hub_download

from segment_anything import sam_model_registry, SamPredictor

from PIL import Image, ImageOps

def FlowSeg(ImgDIR: str, Prompt: str):
    """
    Process images in a directory based on a given prompt.
    
    Args:
        ImgDIR (str): Directory path containing images
        Prompt (str): Prompt for image segmentation
    """
    #Define the repository and filenames for the model and configuration
    repo_id="ShilongLiu/GroundingDINO"
    filename="groundingdino_swinb_cogcoor.pth"
    config_filename="GroundingDINO_SwinB.cfg.py"
    device='cuda'

    #Download the configuration file and model checkpoint from Hugging Face Hub
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=config_filename)

    #Build the model using the downloaded configuration and load the checkpoint
    args = SLConfig.fromfile(cache_config_file)
    Dmodel = build_model(args)
    args.device = device

    #Download the model checkpoint and load it into the model
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = Dmodel.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)

    print("Check the latest .pth from: https://github.com/IDEA-Research/GroundingDINO/releases")

    # Image transformation
    image_source, image = load_image(ImgDIR)
    MainImage=Image.open(ImgDIR).convert('RGB')

    InputImage = ImageOps.expand(MainImage,border=300,fill='white')

    #read the image and get its dimensions
    height, width, depth = np.array(MainImage).shape

    #Setup GroundingDINO for inference and get the bounding boxes, logits, and phrases based on the input image and prompt
    boxes, logits, phrases = dino_predict(model=Dmodel,
                                        image=image,
                                        caption=Prompt,
                                        box_threshold=0.3,
                                        text_threshold=0.3,
                                        device='cuda')
    
    #Print the dimensions of the image and the coordinates of the bounding box
    print(f"Height: {height}")
    print(f"Width: {width}")

    #Convert the bounding boxes to a numpy array and extract the coordinates for the prompt
    NumpyBox = boxes.squeeze().cpu().numpy()

    #Get the X and Y coordinates for the prompt based on the bounding box and image dimensions
    XCord = int(width*NumpyBox[0])
    YCord = int(height*NumpyBox[1])

    print(f"X: {XCord}") 
    print(f"Y: {YCord}")      
    torch.cuda.empty_cache()

    #Load the SAM model and set it up for inference, then get the masks based on the input image and the coordinates from the bounding box
    sam_checkpoint = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    #create the SAM model and load the checkpoint, then move the model to the specified device
    checkpoint = sam_checkpoint
    sam_model = sam_model_registry[model_type]()
    state_dict = torch.hub.load_state_dict_from_url(checkpoint)
    sam_model.load_state_dict(state_dict, strict=True)
    sam_model.to(device=device)

    #create a predictor for the SAM model and set the input image, then predict the masks based on the coordinates from the bounding box
    predictor=SamPredictor(sam_model)

    #image is inputted into the predictor
    predictor.set_image(np.array(InputImage))

    #mask is predicted based on the coordinates from the bounding box, with a point label of 1 and multimask output set to False
    masks, _, _ = predictor.predict(
        point_coords=np.array([[XCord+300, YCord+300]]), 
        point_labels=np.array([1]),
        multimask_output=False
    )
    print(f"Mask shape: {masks.shape}")
    torch.cuda.empty_cache()


    print(masks[0].shape)

    FirstMask = masks[0]
    print(FirstMask.shape)

    CroppedMask=FirstMask[300:height+300, 300:width+300]
    print(CroppedMask.shape)

    return CroppedMask