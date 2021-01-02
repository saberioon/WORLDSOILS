[![ ](https://img.shields.io/badge/version-v0.1--beta.3-blue)](https://github.com/saberioon/WORLDSOILS) [![Build Status](https://travis-ci.com/saberioon/WORLDSOILS.svg?token=vEEN7szXM6Vgp1qaWpuH&branch=main)](https://travis-ci.com/saberioon/WORLDSOILS)





Check the [Wiki page](https://github.com/saberioon/WORLDSOILS/wiki) for Documentation.

Check the [Project tab](https://github.com/saberioon/WORLDSOILS/projects/1) for tracking tasks.

Deadline for delivering model's outputs: __15 Jan 2021__

Tips:

* Push predicted-observed values in CSV format.

* The model’s results (RMSE, R-square, RPD) in any text format. 

* No need to upload any plot. 

* If you change the FNN’s architecture, push it with different name as well.  



To download database and Source code/datasets you can use :

```bash
git clone https://github.com/saberioon/WORLDSOILS.git
```

datasets are available in : data folder 

To setup environment, first install mini anaconda or anaconda then using following code in your terminal:

```bash
conda env create --file environment.yml
```

To activate your virtual environment use this :

```bash
source activate DL2
```

 

__*merging_data.py*__:

merging two csv files based on their ID 

```bash
python src/merging_data.py clhs_lucas15.csv S2a_resampled_lucas15.csv
```

__*fnn.py*__:

Modelling  LUCAS or BSSL data using FNN

```bash
python src/fnn.py -i /path/to/input_file -o /path/to/output_folder -l 3 -e 400 -b 20 
```

-h: help 

-l : number of hidden layer ; default = 3

-e : epoch size; default = 400

-b : batch size; default = 20 

-d : droup out ; default = 0.2





__*LUCAS15_analysis.py*__:

Modelling  LUCAS or BSSL data using FNN

```bash
python src/LUCAS15_analysis.py -i /path/to/input_file -o /path/to/output_folder
```

to see other keys use this :

```bash
python src/LUCAS15_analysis.py -h
```

 



__*metrics.py*__:

calculating R-square, RPD, MSE, RMSE 



__*submitJOB.sh*__:

Submit jobs to cluster 



Updated: 2 Jan 2021

