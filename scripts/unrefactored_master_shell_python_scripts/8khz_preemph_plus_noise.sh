#!/bin/bash


echo " ============================ 8 khz PREEMPH =============================== "

#NO NOISE
for file in GEN/*.mgc ; do

  echo " ================= NO NOISE ============= "
  echo $file
  bare=${file::-4}

  python2.7 ~/Documents/PROJECT_FILES/test_recordings/overlap_and_add_master.py -i $bare -s 8000 -d 250 -p > log_out.txt

done

#NOISY
for file in GEN/*.mgc ; do

  echo " ================= NOISE ============= "
  echo $file
  bare=${file::-4}

  python2.7 ~/Documents/PROJECT_FILES/test_recordings/overlap_and_add_master_PLUS_NOISE_DRAFT_3_DIFF.py -i $bare -s 8000 -d 250 -p -diff > out.txt

done
