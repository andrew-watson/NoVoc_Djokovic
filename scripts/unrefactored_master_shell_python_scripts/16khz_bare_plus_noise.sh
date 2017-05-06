#!/bin/bash


echo " ============================ 16 khz bare =============================== "

#NO NOISE
for file in GEN/*.mgc ; do

  echo " ================= NO NOISE ============= "
  echo $file
  bare=${file::-4}

  python2.7 ~/Documents/PROJECT_FILES/test_recordings/overlap_and_add_master.py -i $bare -s 16000 -d 550 > log_out.txt

done

#NOISY
for file in GEN/*.mgc ; do

  echo " ================= NOISE ============= "
  echo $file
  bare=${file::-4}

  python2.7 ~/Documents/PROJECT_FILES/test_recordings/overlap_and_add_master_PLUS_NOISE_DRAFT_3.py -i $bare -s 16000 -d 550

done
