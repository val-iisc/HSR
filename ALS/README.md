# Attribute Linearity Score

ALS is a novel metric used to measure the linearity of the latent space with respect to individual image attributes.
 
Computing ALS scores requires the docker image `tensorflow/tensorflow:1.15.5-gpu-py3`.

To compute ALS scores, do the following:

- Generate a set of images from the GAN, and for each image, generate the sequence of images after interpolating the latent space.
This can be achieved using [StyleSpace](https://github.com/betterze/StyleSpace) or your own implementation.

- Concatenate all images of each sequence along their width.

- A sample image of this set should look like the one shown below:

<div class="img">
<p align="center">
<img src='../docs/als_sample.png' align="center" width=800>
</p>
</div>

- Run the following from within the aforementioned docker image. The script will generate a single pickle file containing necessary data for computing ALS:

```bash
$ python get_attr.py --img_path /path/to/generated/dataset --save_path /path/to/result/pkl --classifier_path / path/to/attribute/classifiers
``` 

Once the pickle file has been generated, provide `pkl_path`, `save_path_prefix`, and `save_path_prefix_graphs` in `compute_als.py`.

- Run the following from within the aforementioned docker image::

```bash
$ python compute_als.py
``` 

