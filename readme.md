## DaNet 3D Human Pose (the repo is still in progress)

This repository serve as a baseline (for me) code for making 3D human pose model using CNN.
I'm still looking the right model

### Dataset

This repository provide the [Human3.6M](http://vision.imar.ro/human3.6m/description.php) dataset  in 3d points, camera parameters to produce ground truth 2d detections
Unfortunatly we dont provide the stacked hourglass dataset, you will need to download it by your own.

How to download the dataset:

`bash download_data.sh`


### Dependencies

* [h5py](http://www.h5py.org/)
* [tensorflow, keras](https://www.tensorflow.org/) 1.0 or later
* [numpy]



### Training the model

`python Training.py`

### Testing the model
`python Testing.py`

### Example

![Alt text](Examples/test_image.jpeg?raw=true "2D POSE")
![Alt text](Examples/3d_pose.jpeg?raw=true "3D POSE")

### Citing

If you use our code, please cite our work

```
@inproceedings{DaNet 3D Human Pose,
  title={A simple CNN Architecture for 3D human pose},
  author={Ondonda Prince.},
  year={2019}
}