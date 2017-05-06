#!/bin/bash

#Expecting Directory Format:
#Filter_bands_models
    #0_1_GEN
    #0_1_ref_data
    #etc
echo " ============================ 16 khz FREQUENCYBANDS =============================== "
#NO NOISE
for file in ./16khz_0_1k/GEN/*.mgc ; do
#
  echo " ================= NO NOISE ============= "
  echo $file
  bare=${file:16:-4}
  echo $bare
#
  python2.7 ~/Documents/PROJECT_FILES/test_recordings/overlap_and_add_master_4_band.py -1k ./16khz_0_1k/GEN/$bare -2k ./16khz_1k_2k/GEN/$bare -3k ./16khz_2k_3k/GEN/$bare -4k ./16khz_3k_4k/GEN/$bare -s 16000 -d 550 > 4_band.txt

done

#NOISY
for file in ./16khz_0_1k/GEN/*.mgc ; do

  echo " ================= NOISE ============= "
  echo $file
  bare=${file:16:-4}
  echo $bare

  python2.7 ~/Documents/PROJECT_FILES/test_recordings/overlap_and_add_master_4_band_NOISE.py -1k ./16khz_0_1k/GEN/$bare -2k ./16khz_1k_2k/GEN/$bare -3k ./16khz_2k_3k/GEN/$bare -4k ./16khz_3k_4k/GEN/$bare -s 16000 -d 550 > 4_band_plus_NOISE.txt

done
