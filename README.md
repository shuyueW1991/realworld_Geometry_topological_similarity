# Realworld Geometry topological similarity
- why?
    Sometimes, it is likely that we are interested in comparing the different topologies that belongs to entities that we observe in the real world. There are a lot of topologies that is hard to be named simply via `rectangle`, `cubic `, and there is a lack of measuring approach. This repository tries to solve this problem so that you can take some photos of two entities and calculate their delicate topological differences.


## Take photos of a certain entity
- Shoot your camera towards the entity, trying to capture every profile you can get by surrounding it. The number of photos per entity depends on the delicacy and size of geometry. For example, if you are shooting a plank, then maybe a shot can do; but if it is a mosaic that you shoot, then make it more than one.
- Make sure the entity occupy at least 60% of the image 
- Make sure successful focus on the entity's surface
- As for metallic surface, make sure the reflective light that affect the depth estimation in subsequent point cloud reconstruction.

## Pipe your photos through COLMAP
- installation of COLMAP
    -  mac: 
        - brew install colmap
        - if in mainland People's Republic, change your brew source to tsinghua
- colmap gui
    - follow most of the operations in this [blog](https://zhuanlan.zhihu.com/p/576416530)
    - the format resembles those in [360_v2](https://jonbarron.info/mipnerf360/), [nerf_llff_data](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7) and [LERF](https://drive.google.com/drive/folders/1vh0mSl7v29yaGsxleadcj-LCZOE_WEWB?usp=sharing).
    - The final data folderwise structural goes like:

        ```
        ./data
            /<object_name>
                /images
                /sparse
            ...
        ```


## Deploy thie repository and install all the necessary.






















## Acknowledgement
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting.git)
- [SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians.git)
- [COLMAP_LLFF](https://zhuanlan.zhihu.com/p/576416530)

