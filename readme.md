## DaNet 3D Human Pose (the repo is still in progress)

This repository serve as a baseline (for me) code for making 3D human pose model using CNN.
I'm still looking the right model.
The camera orientation doesnt work as expected
Warning: this repo is heavily inspired by the Facebook Video3D repo.

### Dataset

This repository provide the [Human3.6M](http://vision.imar.ro/human3.6m/description.php) dataset  in 3d points, camera parameters to produce ground truth 2d detections.

How to download the dataset:

`bash download_data.sh`


### Dependencies

* [h5py](http://www.h5py.org/)
* [tensorflow, keras](https://www.tensorflow.org/) 1.0 or later
* [numpy]


### Generate the data
`python3 Utils/PrepareDataset.py` 
after the dataset prepation has done , please place the files `data_2d_h36m_gt.npz` and `data_3d_h36m.npz` in the data directory

### Training the model
`python3 Training.py`

### Testing the model
`python3 Testing.py`

### Example

![Alt text](Examples/test_image.jpeg?raw=true "2D POSE")
![Alt text](Examples/figure_3d.png?raw=true "3D POSE")

### Citing

If you use our code, please cite our work

```
@inproceedings{DaNet 3D Human Pose,
  title={A simple CNN Architecture for 3D human pose},
  author={Ondonda Prince.},
  year={2019}
}