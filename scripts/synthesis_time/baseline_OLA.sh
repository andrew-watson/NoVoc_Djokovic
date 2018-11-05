#!/bin/bash


echo " ============================ Performing Overlap and Add =============================== "

for file in mgc/*.mgc ; do

  echo $file
  bare=${file::-4}

  python2.7 NoVoc_Djokovic/scripts/REFACTOR_overlap_and_add_master.py -i $bare -s 16000 -d 250 > log_out.txt

done

