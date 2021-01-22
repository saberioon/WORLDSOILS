#!/usr/bin/bash

# SPDX-FileCopyrightText:`2020 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam, Germany'
# SPDX-License-Identifier: MIT

# submit single python job to the GFZ cluster
# written by : Mohammadmehdi Saberioon - revised date: 14.12.2020



dir_out="/home/saberioo/WORLDSOILS/script-out"
dir_in="/home/saberioo/WORLDSOILS/data/CHIME_merged_lucas15.csv"

# shellcheck disable=SC2006
i=`date +"%s"`

#program_input="some options for the python script"

  # create run script
  #  program_input="x"
  # settings and commands for running my job
  echo "source /home/saberioo/soft/condaroot/conda-main/etc/profile.d/conda.csh" > /home/saberioo/WORLDSOILS/run_jobs/run_"$i".tmp
  echo "conda activate DL2" >> /home/saberioo/WORLDSOILS/run_jobs/run_"$i".tmp
  echo "python /home/saberioo/WORLDSOILS/src/LUCAS15_analysis.py -i $dir_in -o $dir_out" -b 40 -e 500 >> /home/saberioo/WORLDSOILS/run_jobs/run_"$i".tmp
  # show what will be done
  echo "python /home/saberioo/WORLDSOILS/src/LUCAS15_analysis.py -i $dir_in -o $dir_out" -b 40 -e 500

  # submit run script to cluster
  bsub -R "rusage[mem=100]" -o out/run_"$i".out -e out/run_"$i".err -q qintel -J job_"$i" csh /home/saberioo/WORLDSOILS/run_jobs/run_"$i".tmp
  echo "sent job $i"