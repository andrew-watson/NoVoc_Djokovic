#!/bin/bash


echo " ======== Performing Overlap and Add ======== "

for file in mgc/*.mgc ; do

  echo $file
  bare=${file%.*}
  echo $bare

  python2.7 ./scripts/synthesis_time/REFACTOR_overlap_and_add_master.py -i $bare -s 16000 -d 550 > log_out.txt

done


