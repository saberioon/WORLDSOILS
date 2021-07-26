



## WORLDSOILS - Fully Connected Neural Network

[![ ](https://img.shields.io/badge/version-v0.1--beta.7-blue)](https://github.com/saberioon/WORLDSOILS) [![Build Status](https://travis-ci.com/saberioon/WORLDSOILS.svg?token=vEEN7szXM6Vgp1qaWpuH&branch=main)](https://travis-ci.com/saberioon/WORLDSOILS) [![DOI](https://zenodo.org/badge/318167857.svg)](https://zenodo.org/badge/latestdoi/318167857)    [![DOI](https://img.shields.io/badge/packaging-poetry-299bd7?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAASCAYAAABrXO8xAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAJJSURBVHgBfZLPa1NBEMe/s7tNXoxW1KJQKaUHkXhQvHgW6UHQQ09CBS/6V3hKc/AP8CqCrUcpmop3Cx48eDB4yEECjVQrlZb80CRN8t6OM/teagVxYZi38+Yz853dJbzoMV3MM8cJUcLMSUKIE8AzQ2PieZzFxEJOHMOgMQQ+dUgSAckNXhapU/NMhDSWLs1B24A8sO1xrN4NECkcAC9ASkiIJc6k5TRiUDPhnyMMdhKc+Zx19l6SgyeW76BEONY9exVQMzKExGKwwPsCzza7KGSSWRWEQhyEaDXp6ZHEr416ygbiKYOd7TEWvvcQIeusHYMJGhTwF9y7sGnSwaWyFAiyoxzqW0PM/RjghPxF2pWReAowTEXnDh0xgcLs8l2YQmOrj3N7ByiqEoH0cARs4u78WgAVkoEDIDoOi3AkcLOHU60RIg5wC4ZuTC7FaHKQm8Hq1fQuSOBvX/sodmNJSB5geaF5CPIkUeecdMxieoRO5jz9bheL6/tXjrwCyX/UYBUcjCaWHljx1xiX6z9xEjkYAzbGVnB8pvLmyXm9ep+W8CmsSHQQY77Zx1zboxAV0w7ybMhQmfqdmmw3nEp1I0Z+FGO6M8LZdoyZnuzzBdjISicKRnpxzI9fPb+0oYXsNdyi+d3h9bm9MWYHFtPeIZfLwzmFDKy1ai3p+PDls1Llz4yyFpferxjnyjJDSEy9CaCx5m2cJPerq6Xm34eTrZt3PqxYO1XOwDYZrFlH1fWnpU38Y9HRze3lj0vOujZcXKuuXm3jP+s3KbZVra7y2EAAAAAASUVORK5CYII=)](https://python-poetry.org/) 



Check the [Wiki page](https://github.com/saberioon/WORLDSOILS/wiki) for Documentation.

Check the [Project tab](https://github.com/saberioon/WORLDSOILS/projects/1) for tracking tasks.  



To download database and Source code/datasets you can use :

```bash
git clone https://github.com/saberioon/WORLDSOILS.git
```

datasets are available in : data folder 

To setup environment, first install [poetry](https://python-poetry.org/)  then using following code in your terminal:

```bash
poetry install
```

To activate your virtual environment use this :

```bash
poetry shell
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

-r : learning rate ; default = 0.01

-k : kernel initializer ; default = he_normal



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



__***outlier_finder.py***__:

Finding outliers using Mahalanobis distance + Principal Component Analysis 



Updated: 26 July 2021.

