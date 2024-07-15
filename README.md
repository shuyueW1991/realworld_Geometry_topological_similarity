# Realworld Geometry topological similarity
## Why?
- Sometimes, it is likely that we are interested in comparing the different topologies that belongs to entities that we observe in the real world. There are a lot of topologies that is hard to be named simply via `rectangle`, `cubic `, and there is a lack of measuring approach. 
- This repository tries to solve this problem so that you can take some photos of two entities and calculate their delicate topological differences.


## Take photos of a certain entity
- Shoot your camera towards the entity, trying to capture every profile you can get by surrounding it. The number of photos per entity depends on the delicacy and size of geometry. For example, if you are shooting a plank, then maybe a shot can do; but if it is a mosaic that you shoot, then make it more than one.
- Make sure the entity occupy at most 50% of the image, otherwise the automatic mask generator of SAM will give too detailed segmentations. 
- Make sure successful focus on the entity's surface.
- As for metallic surface, make sure the reflective light that affect the depth estimation in subsequent point cloud reconstruction.

## Pipe your photos through COLMAP
- installation of COLMAP
    -  mac: 
        - brew install colmap
        - if in mainland People's Republic, change your brew source to tsinghua
- colmap operation
    - follow most of the operations in this [blog](https://zhuanlan.zhihu.com/p/576416530)
    - the format resembles those in [360_v2](https://jonbarron.info/mipnerf360/), [nerf_llff_data](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7) and [LERF](https://drive.google.com/drive/folders/1vh0mSl7v29yaGsxleadcj-LCZOE_WEWB?usp=sharing).
    - create a folder of your things under <data>.
    - create <images> under the above folder, and put all the images into <images>.
    - bash input `colmap gui` (or other way of opening colmap according to your system and installation method).
    - Click `File` --> `New project`, create `databsase.db` parallel to <images>.
    - select Images as `images`.
    - Then, `Processing` --> `Feature extraction` where camera model being `simple_pinhole` in most cases.
    - Then, `Processing` --> `Feature matching`.
    - Then, `Reconstruction` --> `Start reconstruction`. 
    - Then, `file` --> `export model` in a manually created folder <sparse/0/> parallel to <images>.
    - The final data folderwise structural goes like:
        ```
        ./data
            /<object_name>
                /images
                /sparse
                    /0
                database.db
            ...
        ```


## Deploy thie repository and install all the necessary.
My remoter is operated on Ubuntu22.04, with Cuda version being 12.1.

Clone this repository:
···bash
git clone https://github.com/shuyueW1991/realworld_Geometry_topological_similarity.git
···
Then install the dependencies:
```bash
conda env create --file my_env.yml
conda activate feature3dgs_2_seganygaussians_v
```
The <submodules> needs be pip installed locally. You may refer yourself to the repo of <gaussian-splatting>.



## Pre-train the Gaussians via 3dgs
```bash
python train_scene.py -s <path to dataset >
```
You can prepend it with `CUDA_VISIBLE_DEVICES=x` to select your gpu.
```bash
CUDA_VISIBLE_DEVICES=1 python train_scene.py -s data/lying_zero/
```

The trained result is in folder <output> with a unique number in the foldername.


## Get the sam_masks and corresponding mask scales
```bash
python extract_segment_everything_masks.py --image_root <path to the scene data> --sam_checkpoint_path <path to the pre-trained SAM model> --downsample <1/2/4/8>
```
Now we have automatically generatened masks that covers all over the entire iamges.


## Group the segmentation semantically

For each pixel in the image, it corresponds to three mask of diffrerent scale  in defaultly configured SAM automatic mask generator.
The hierarchy of masks in the sense of semantcis needs be constructed.

Then a great masterpiece comes out:[GARField](https://arxiv.org/abs/2401.09419). It says:

> ...given an input image set, we extract a set of candidate groups by densely querying SAM, and assign each a physical scale by deprojecting depth from the NeRF. These scales are used to train a scale-conditioned affinity field (Right). During training, pairs of sampled rays are pushed apart if they reside in different masks, and pulled together if they land in the same mask. Affinity is supervised only at the scale of each mask, which helps resolve conflicts between them.

For a given point in 3-dimensional space, give it a scale, you then get a feature vector. That is to say a feature vector is determined by xyz and its scale. The feature vector is result from training such that in different masks, the feature-vectorwise distance is big and otherwise small. Therefoe, the trend is, with bigger scale the feature vectors are closer. The figure 3 in their paper tells the whole story. You should read it!


The next step preps for capturing the affinity between the features.

```bash
python get_scale.py --image_root <path to the scene data> --model_path <path to the pre-trained 3DGS model>
```

like,

```bash
CUDA_VISIBLE_DEVICES=1 python extract_segment_everything_masks.py --image_root data/lying_zero  --downsample 8
```
(Note, if the picture is very big in size, the problem will come up as runtimerror shit. Maybe it is just a gpu limited memory issue. Just manually downsize the images and rename the new images set with 'images_' + downsizing_times.)

```bash
CUDA_VISIBLE_DEVICES=1 python get_scale.py --image_root data/lying_zero --model_path output/271e0e7d-2
```    
(Note that the get_scale.py is now fixed at _8 downsized images. )


Optionally, if you want to try the open-vocabulary segmentation, extract the CLIP features first:
```bash
python get_clip_features.py --image_root <path to the scene data>
```

like,

```bash
CUDA_VISIBLE_DEVICES=1 python get_clip_features.py --image_root data/lying_zero
```













## Acknowledgement
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting.git)
- [SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians.git)
- [COLMAP_LLFF](https://zhuanlan.zhihu.com/p/576416530)

