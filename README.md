![https://github.com/saberioon/WORLDSOILS/releases](https://img.shields.io/badge/version-v0.1--beta.2-blue)



Check the [Wiki page](https://github.com/saberioon/WORLDSOILS/wiki) for Documentation.

Check the [Project tab](https://github.com/saberioon/WORLDSOILS/projects/1) for tracking tasks.

Deadline for delivering model's outputs: __15 Jan 2021__

Tips:

* Datasets are stored under __"compressed"__ folder . 

* Push predicted-observed values in CSV format.

* The model’s results (RMSE, R-square, RPD) in any text format. 

* No need to upload any plot. 

* If you change the FNN’s architecture, push it with different name as well.  



To download database and Source code you can use :

```
git clone https://github.com/saberioon/WORLDSOILS.git
```

> This is a private repository, therefore you need to enter your GitHub's password  

__*merging_data.py*__:

merging two csv files based on their ID 

```bash
python merhing_data.py clhs_lucas15.csv S2a_resampled_lucas15.csv
```



__*LUCAS15_analysis.py*__:

Modelling  LUCAS or BSSL data using FNN

```
python LUCAS15_analysis.py /path/to/input_file /path/to/output_folder
```

 

__*metrics.py*__:

calculating R-square, RPD, MSE, RMSE 



__*submitJOB.sh*__:

Submit jobs to cluster 



Updated: 15 Dec 2020

