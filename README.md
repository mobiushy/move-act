<div align="center">
<h2>Move and Act: Enhanced Object Manipulation and Background Integrity for Image Editing</h2>
</div>

### Setup
You can build the environment as follow: 
```
conda env create -f environment.yaml
conda activate ldm
```
 
### Usage
In order to edit the action of an object in an image and specify the generation position of the edited object, you can use `run_mna.py`. For example:
```
python run_mna.py --img_path './data/0040.jpg' --cond_path './condition/0040.json'
```
Or you can specify that the edited object is generated at its original location:
```
python run_mna.py --img_path './data/0012.jpg' --cond_path './condition/0012.json'
```

The direcory structure of editing results are as follows:
```
outputs/
|-- text prompt/
|   |-- 42.png 
|   |-- 42_bbox.png 
|   |-- 1.png
|   |-- 1_bbox.png 
|   |-- ...
```
