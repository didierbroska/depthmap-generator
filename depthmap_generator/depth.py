"""
Compute depth maps for images in the input folder.
"""

import logging
import os
from glob import glob

import cv2
import numpy as np
import torch

from .monodepth_net import MonoDepthNet as Net
from .utils import read_image, resize_depth, resize_image, write_depth


class Depthmap:
    """
    Class generates depth map thru MonoDepthNN.
    """

    def __init__(
        self, img_folder: str, output: str, model: str, device: str = "cpu"
    ):
        """
        Init Depthmap object.
        Arguments:
            img_folder (str) : Images folder path
            ouput (str) : Path folder where maps stored
            model (str) : Path to the monodepth model.
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # Inits
        self.__init_device(device)
        self.__load_nn(model)
        self.__get_images(img_folder)
        self.__mk_output_fld(output)

    def __init_device(self, device: str):
        """
        Init device used by PyTorch

        Arguments:
            device (str) : type of device used.
        """
        logging.info("[INFO] - Initializing ...")
        self.device = torch.device("cpu")
        logging.debug(f"[DEBUG] - Device using is {device}.")

    def __load_nn(self, model: str):
        """
        Load Neural Network in device.

        Arguments:
            model (str) : Path to the monodeth model.
        """
        logging.info("[INFO] - Load Neural Network")
        if not os.path.isfile(model):
            raise FileNotFoundError("Model file not found !")
        self.model = Net(model)
        self.model.to(self.device)
        self.model.eval()

    def __get_images(self, img_folder: str):
        """
        Get images from path folder.

        Arguments:
            img_folder (str) : Path to images folder.
        """
        self.imgs = glob(os.path.join(img_folder, "*"))
        if len(self.imgs) == 0:
            raise FileNotFoundError("Images folder is empty !")

    def __mk_output_fld(self, output: str):
        """
        Make folder for output deth maps.

        Arguments:
            output (str) : Output path for deth maps.
        """
        os.makedirs(output, exist_ok=True)
        self.output = output

    def generate(self):
        """
        Run to generate deth maps.
        """
        logging.info("[INFO] - Start processing")

        for i, img_path in enumerate(self.imgs):
            logging.info(
                "[INFO] - Processing {} ({}/{})".format(
                    os.path.basename(img_path), i + 1, len(self.imgs)
                )
            )

            # Convert image in NP array
            img = read_image(img_path)

            # Resizing image and precompute
            scale = 640.0 / max(img.shape[0], img.shape[1])
            target_height, target_width = (
                int(round(img.shape[0] * scale)),
                int(round(img.shape[1] * scale)),
            )
            img_input = resize_image(img)
            img_input = img_input.to(self.device)

            # Computing
            with torch.no_grad():
                out = self.model.forward(img_input)

            depth = resize_depth(out, target_width, target_height)
            img = cv2.resize(
                (img * 255).astype(np.uint8),
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )

            # Build path and save deth map image
            filename = os.path.join(
                self.output, os.path.splitext(os.path.basename(img_path))[0]
            )
            write_depth(filename, depth, bits=2)

        logging.info("[INFO] - Finished.")
