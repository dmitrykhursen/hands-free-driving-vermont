# Hands-Free Driving Vermont: Road-Scene Image Analysis

A dashcam image and video annotator. It detects road objects with Faster R-CNN, sends detected traffic signs to a second CNN that classifies the sign type, and runs classical lane detection on highway footage.

**1st place, Advanced Machine Learning Projects, UVM Computer Science Fair 2022.**

<div align="center">
  <img src="images/2.jpg" width="600" alt="Annotated street scene">
</div>

## Context
Final project for **CS 254 Machine Learning** at the University of Vermont (Fall 2022), built by a team of four during my exchange semester. This repository copies the team's original private repository with its full commit history, so every teammate's commits are kept and attributed.

## How it works
1. **Object detection:** Faster R-CNN (PyTorch, through [detecto](https://github.com/alankbi/detecto)) trained on [Berkeley DeepDrive (BDD100K)](https://www.vis.xyz/bdd100k/) labels for 10 road classes: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign. Boxes above 0.85 confidence are kept and drawn in red.
2. **Traffic-sign classification:** each "traffic sign" box is cropped, padded and resized to 32×32 RGB, then classified by a Keras CNN into 8 classes: stop, yield, speed limit, pedestrian crossing, do not enter, school zone, signal ahead, other. If the CNN is more than 0.7 confident, the box turns blue and shows the sign type.
3. **Lane detection (OpenCV):** HLS colour masks for white and yellow lines, Canny edges, a region-of-interest mask and a probabilistic Hough transform. It only gave usable results on the highway videos. Faded lines and winding city roads broke it.

<div align="center">
  <img src="images/3.jpg" width="600" alt="Detected objects and classified signs">
</div>

## Results
- **Sign classifier:** about 90% validation accuracy on roughly 1,000 images, from the LISA Traffic Sign Dataset plus our own photos of Vermont signs.
- **Detector:** we did not evaluate it quantitatively. IoU/mAP evaluation wasn't finished before the deadline, so the detector was judged on video only.
- **Demo:** [video on YouTube](https://youtu.be/kuKiGpaKpwA). It covers our own dashcam footage from downtown Burlington plus BDD clips in daylight, at night, in rain, and on the highway with lane detection.

[![Demo video](images/1.jpg)](https://youtu.be/kuKiGpaKpwA)

## Award: UVM CS Fair 2022
We entered the project in the 2022 University of Vermont Computer Science Fair and won **1st place in the Advanced Machine Learning Projects bracket** ([fair results](https://csfair.w3.uvm.edu/2022)). The prize was a team result.

<div align="center">
  <img src="images/csfair.jpg" width="500" alt="UVM CS Fair 2022">
</div>

## Limitations
This was an introductory ML course project, and the code shows it:
- There are no tests and no metrics code in the repo.
- The detector weights and the datasets are not included.
- The sign CNN's training code is still the earlier grayscale version, but the deployed model is RGB.

## Team and contributions
| Member | Main contribution |
|---|---|
| Shawn Rodgers | Faster R-CNN object detection (`lib/object_detection.py`), the `run.py` pipeline, project setup and the original README |
| David Smith | Lane detection (`lib/lane_detection.py`) |
| **Dmytro Khursenko** | Traffic-sign classification stage (details below) |
| Harrison Thompson | Team member |

**My part: traffic-sign classification**
- Wrote `lib/sign_classification.py`: data loading, a 60/20/20 train/val/test split, the CNN (2× Conv 5×5 → pool → 2× Conv 3×3 → pool → Dense 500 → Dropout 0.5 → softmax), and training and inference.
- Built the dataset from LISA signs plus self-collected Vermont sign photos, and trained several model versions.
- Switched the model from grayscale to RGB input, since colour separates stop, yield and school-zone signs. Fixed a colour-encoding bug in the pipeline that hurt sign recognition.
- Integrated the classifier into `run.py` as a second stage on the detector's sign crops, with its own confidence threshold.

## Tech stack
Python, PyTorch + detecto (Faster R-CNN), TensorFlow/Keras (sign CNN), OpenCV, scikit-learn, NumPy, Matplotlib, imageio.

## How to run
```bash
pip install -r requirements.txt
python setup.py                      # creates data/, models/, input/, output/ folders
python run.py 1 path/to/image.jpg    # single image
python run.py 2 path/to/images/      # directory of images
python run.py 3 path/to/video.mp4    # video
python run.py 4                      # train the detector (needs BDD100K in data/)
```
The sign model `models/RGB_eightClasses_with_VT_data.h5` is included. The Faster R-CNN weights (`models/faster_rcnn_5000_instances_12_epochs.pth`) are not, so you need to train them or add them yourself. Flags and thresholds are in `config.py`.

## Repository structure
```
run.py                      # CLI driver: image / directory / video / training
config.py                   # flags, class map, thresholds, model paths
setup.py                    # creates the folder layout
lib/object_detection.py     # Faster R-CNN training and inference
lib/sign_classification.py  # traffic-sign CNN
lib/lane_detection.py       # classical lane detection
models/                     # sign-classifier weights
images/                     # README screenshots
```

## License
MIT, see [LICENSE](LICENSE). Copyright (c) 2023 Shawn Rodgers, David Smith, Dmytro Khursenko.

## Contact
**Dmytro Khursenko**: [dmytro.khursenko@gmail.com](mailto:dmytro.khursenko@gmail.com) · [LinkedIn](https://www.linkedin.com/in/dmytrokhursenko/) · [GitHub](https://github.com/dmitrykhursen)
