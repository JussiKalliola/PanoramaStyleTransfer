# Style Transfer

## Prerequisites
- Python 3.7
- [PyTorch 1.12.0](http://pytorch.org/)
- [NumPy](http://www.numpy.org/)
- [PIL](http://pillow.readthedocs.io/en/3.1.x/installation.html)

## Datasets
Download [Coco Dataset](https://cocodataset.org/#download) - 2014 Train images [83K/13GB]

Download [Panorama Dataset](https://www.dropbox.com/s/skxqj94rno9ihjq/360-Dataset.zip?dl=0)

## Usage
### Train

You can train a model for a given style image with the following command:

```bash
$ python style.py train --style-image "path_to_style_image" --dataset "path_to_coco"
```

Here are some options that you can use:
* `--gpu`: id of the GPU you want to use (if not specified, will train on CPU)
* `--visualize`: visualize the style transfer of a predefined image every 1000 iterations during the training process in a folder called "visualize"

So to train on a GPU with mosaic.jpg as my style image, MS-COCO downloaded into a folder named coco, and wanting to visualize a sample image throughout training, I would use the following command: 

```bash
$ python style.py train --style-image style_imgs/mosaic.jpg --dataset coco --gpu 1 --visualize 1
```

### Evaluation

You can stylize an image with a pretraind model with the following command. Pretrained models for mosaic.jpg and udine.jpg are provided.

```bash
$ python style.py transfer --model-path "path_to_pretrained_model_image" --source "path_to_source_image" --output "name_of_target_image"
```

You can also specify if you would like to run on a GPU:
* `--gpu`: id of the GPU you want to use (if not specified, will train on CPU)

For example, to transfer the style of mosaic.jpg onto maine.jpg on a GPU, I would use:

```bash
$ python style.py transfer --model-path model/1657989431_mosaic.model --source content_imgs/maine.jpg --output maine_mosaic.jpg --gpu 1
```


## Acknowledgements
* This repo is based on code found in [this PyTorch example repo](https://github.com/dxyang/StyleTransfer)
