#!/bin/bash


echo " ============================ 16 khz CENTRED =============================== "

#NO NOISE
for file in GEN/*.mgc ; do

  echo " ================= NO NOISE ============= "
  echo $file
  bare=${file::-4}

  python2.7 ~/Documents/PROJECT_FILES/test_recordings/OLA_master_centre.py -i $bare -s 16000 -d 959 -t 15 > log_out.txt

done

#NOISY
for file in GEN/*.mgc ; do
  
  echo " ================= NOISE ============= "
  echo $file
  bare=${file::-4}

  python2.7 ~/Documents/PROJECT_FILES/test_recordings/OLA_master_centre_NOISE.py -i $bare -s 16000 -d 959 -t 15 > log_out.txt

done
