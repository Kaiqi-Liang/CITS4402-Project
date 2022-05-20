# CITS4402-Project
## Setup
1. Check whether `conda` is installed by running
```
conda env list
```
2. If it says `command not found` go install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. After installing it try running the command again, it should list all the environments in the system, check whether the name `cv` exists
4. If not create a new environment with all the dependencies installed
```
conda env create --file cv.yml
conda activate cv
```
5. Otherwise just use another name that is not taken, for example
```
mv cv.yml name_not_taken.yml
conda env create --file name_not_taken.yml
conda activate name_not_taken
```
6. Run the server
```
python backend/server.py
```
7. Open the file [frontend/index.html](frontend/index.html) in a browser

## Assumptions
### Parser Function
Frame interval is one of the hyperparameters that can be set in the GUI. Default is 1 as is necessary to stabilise the Kalman filter but 10 demonstrated stronger results for candidate detection and discrimination.

### Candidate Discrimination
Pixels clusters of size less than 3 are ignored in region growing. Less than three values is unlikely to be a car, and statistical calculations on such a small set of values is illogical.

Morphological cues are calibrated experimentally, in particular by plotting the distribution of area, extent, major axis and eccentricity. Thresholds for each cue are determined based on the bell curve plotted.

Intersection of Union metric is set to 0.3, as any higher allows no matches through our algorithm. We acknowledge that this is lower than expected for a matching metric, and many not be a good representation of how many true matches are found.

### Kalman Filter
The state vector for each track is initialized based on the cluster's centroid and with velocities and accelerations of zero. Covariance matrix of the estimated is initialized to be the covariance matrix of the motion model. The standard deviation of the position estimate is set equal to 1 as we assume this estimation, based on previous candidate detection and discrimination, is reasonable. However, the standard deviation for the velocities and accelerations is set higher to 4 as 0 is not a reasonable assumption and we have no way of knowing what the object's speed and acceleration is.

Cost of non-assignment, that is the cost of assigning a pseudo track to a hypothesis or a pseudo hypothesis to a track, is a positive hyperparameter that can be set in the GUI. If no cost is provided, the default cost is set to slightly larger than the mean of euclidean distances calculated in track association. This is to ensure that the probability of non-assignment is low.

## Acknowledgement
We acknowledge that our Kalman filter is not fully functional. It is currently set to initialize based on the first frame given, and propagate and update states throughout the remaining frames. The unassigned tracks should be passed into the template matching function and if matched then continue through to update and so on. The unassigned hypothesis should be passed into the init tracks function, to initialize their track ID and state vector, and then continue through to the next iteration. These two tangental parts of the Kalman filter are not fully functional. For this reason, the average number of moving objects detected per frame, as a time series, present a descending trend as no unassigned hypothesis are initialized and no unassigned tracks are template matched.