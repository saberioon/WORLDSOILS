#!/usr/bin/bash

# SPDX-FileCopyrightText:`2020 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam, Germany'
# SPDX-License-Identifier: MIT

# submit  python jobs to the GFZ cluster
# written by : Mohammadmehdi Saberioon - revised date: 14.12.2020



dir_out="/home/saberioo/WORLDSOILS/script-out"
dir_in="/home/saberioo/WORLDSOILS/data/cluster_input.txt"
length_file=`wc -l $dir_in | awk '{print $1}'`
data_path="/home/saberioo/WORLDSOILS/data"

for i in `seq 1 1 5` # lines in the cluster_input.txt file that shall be used as input
  do
  # do not submit when queue too full
    no_in_q=` bjobs |wc -l`
    while [ $no_in_q -gt 500 ]
    do
      echo "queue too full: " $no_in_q
      no_in_q=` bjobs |wc -l`
      sleep  30
    done


  # create run script for each job in list
    program_input=`awk '{if (NR=='$i') print $0}' $dir_in`
    # settings and commands for running my job
    echo "source /home/saberioo/soft/condaroot/conda-main/etc/profile.d/conda.csh" > /home/saberioo/WORLDSOILS/run_jobs/run_"$i".tmp
    echo "conda activate DL2" >> /home/saberioo/WORLDSOILS/run_jobs/run_"$i".tmp
    echo "python /home/saberioo/WORLDSOILS/src/fnn.py -i $data_path/$program_input -o $dir_out" -b 20 -e 500 -l 5   >> /home/saberioo/WORLDSOILS/run_jobs/run_"$i".tmp

    # show what will be done
    echo "python /home/saberioo/WORLDSOILS/src/fnn.py -i $data_path/$program_input -o $dir_out" -b 20 -e 500 -l 5

    # submit run script to cluster
    bsub -R "rusage[mem=100]" -o out/run_"$i".out -e out/run_"$i".err -q qintel -J job_"$i" csh /home/saberioo/WORLDSOILS/run_jobs/run_"$i".tmp
    echo "sent job $i"

  done