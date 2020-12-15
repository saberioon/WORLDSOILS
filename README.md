Check the [Wiki page](https://github.com/saberioon/WORLDSOILS/wiki) for Documentation.

Check the [Project tab](https://github.com/saberioon/WORLDSOILS/projects/1) for tracking tasks.

Deadline for delivering model's outputs: __15 Jan 2021__

Tips:

* Datasets are stored under __"compressed"__ folder . 

* Push predicted-observed values in CSV format.

* The model’s results (RMSE, R-square, RPD) in any text format. 

* No need to upload any plot. 

* If you change the FNN’s architecture, push it with different name as well.  





merging_data.py:

merging two csv files based on their ID 

```bash
python merhing_data.py clhs_lucas15.csv S2a_resampled_lucas15.csv
```



LUCAS15_analysis.py:

Modelling  LUCAS or BSSL data using FNN

```
python LUCAS15_analysis.py /path/to/input_file /path/to/output_folder
```

 

metrics.py:

calculating R-square, RPD, MSE, RMSE 



submitJOB.sh:

Submit jobs to cluster 













Updated: 3 Dec 2020

