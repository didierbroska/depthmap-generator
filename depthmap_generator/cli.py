# coding: utf-8
"""
Cli entrypoint for deth map generator from monocular picture.
"""

import argparse
import logging
import os

from . import utils
from .depth import Depthmap

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "model.pt",
)

DEFAULT_FILE_ID = "1Jj0BiRElC--8Q0wekzGhop6Mxh-z0mYi"


def cli_args():
    parser = argparse.ArgumentParser(
        prog="DepthMap-generator",
        description="Depth maps generator from monocular image(s).",
    )
    parser.add_argument(
        "-i",
        "--images_path",
        help="Input path to images sequence",
        type=str,
        required=True,
        metavar="PATH",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output path for depth maps",
        type=str,
        default=os.path.join(".", "depthmaps"),
        metavar="PATH",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="path to model MiDaS if you want use custom",
        default=DEFAULT_MODEL_PATH,
        metavar="PATH",
    )

    parser.add_argument(
        "-d",
        "--device",
        help="device selection for computing. (cpu/cuda/opencl)",
        default="cpu",
        choices=("cpu", "cuda", "opencl"),
        metavar="DEVICE",
    )

    parser.add_argument(
        "-v", "--verbose", help="increase verbose", action="store_true"
    )
    parser.add_argument(
        "-q", "--quiet", help="quiet verbose", action="store_true"
    )

    return parser.parse_args()


def main():
    opts = cli_args()

    # TODO - ameliorer le format de log
    if not opts.quiet:
        if opts.verbose:
            print("verbose")
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    # TODO - other devices
    if opts.device == "cuda" or opts.device == "opencl":
        raise NotImplementedError(
            "This device is not implemented for the moment !"
        )

    logging.debug(
        f'[DEBUG] - Default path for model is "{DEFAULT_MODEL_PATH}"'
    )

    if not os.path.isfile(opts.model) and opts.model == DEFAULT_MODEL_PATH:
        utils.g_download(DEFAULT_FILE_ID, DEFAULT_MODEL_PATH)

    depthmap = Depthmap(opts.images_path, opts.output, opts.model, opts.device)
    depthmap.generate()


if __name__ == "__main__":
    main()
