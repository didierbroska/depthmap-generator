# DepthMap Generator

[![DepthMap Generator in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Lk2hnB6giW3-LPDiMAYNYptvybsqw1En) ![Python versions badge](https://img.shields.io/badge/Python-3.7-blue)


<p align='center'>
    <img src='https://drive.google.com/uc?id=1gL1N5pOj66FYOBQGnPC2JWxI3_ke7_us' width='400' alt='Depth Map Generator Example'/>
</p>

We propose a method for converting a single RGB-D input image into a depth map black and white image.

- Documentation : TODO
- Free Software : MIT License

## Features

- Command line for more information run `depthmap-generator -h`.
- Compatible Windows 10 and GNU/Linux.
- Graphical interface *(comming soon)*.
- Handle lot of images.
- Mutli-Thread with CPU support.
- Cuda support *(comming soon)*.

## Prerequisites

- Linux or Windows
- Poetry or Pip
- Python 3.7+
- PyTorch 1.8.0+

## Quickstart

Clone the project. (this requires `git`)

```bash
git clone https://github.com/didierbroska/depthmap-generator.git
cd depthmap-generator
```

Install dependencies. Project use `poetry` but you can used simply `pip` too.

<!-- TODO - explain virtual env -->

**poetry**

```bash
# Install all deps (prod+dev)
poetry install


# Only prod without dev dependencies
poetry install --no-dev
```

**pip**

```bash
# Install all deps (prod+dev)
pip install -r deps/requirements-dev.txt

# Only prod without dev dependencies
pip install -r deps/requirements.txt
```

Now you can use it and enjoy !

```bash
# poetry
poetry run python depthmap_generator/cli.py -i images/

# pip
python depthmap_generator/cli.py -i images/
```

## License

This work is licensed under MIT License. See [LICENSE](LICENSE) for details.

## Credits

You can found this in [Authors](AUTHORS.md) for details.
