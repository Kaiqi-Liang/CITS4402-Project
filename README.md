# CITS4402-Project
## Set up the environment to run this program
1. Check whether `conda` is installed
```
conda env list
```
2. If it says `command not found` go install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. After installing it try running the command again, it should list all the environments in the system, check if the name `cv` exists
4. If so just use another name that is not taken, for example
```
mv cv.yml name_not_taken.yml
conda env create --file name_not_taken.yml
conda activate name_not_taken
```
5. Otherwise create a new environment with all the dependencies installed
```
conda env create --file cv.yml
conda activate cv
```
6. Run the server
```
python backend/server.py
```
7. Open the file [frontend/index.html](frontend/index.html) in a browser