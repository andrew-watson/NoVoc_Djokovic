#!/bin/bash

# Takes each wav file in a folder, performs pitchmark extraction with REAPER, then for each line in the corresponding .pm file, samples from the range given by the timestamps then passes this through the python file. Output of the python file is printed and tee to an output file for each wav.

#Run from test_recordings directory

cd ./downsamp_wavs

for file in ./*.wav ; do

  #check
  echo $file
  bare=${file::-4}
  new=${file::-4}_d
  echo $bare
  #Make sure downsampled to 48kH
  #check with
  #sox --i ./$file
  #MAYBE BEST TO DO THIS BEFORE AND OUTSIDE SCRIPT
  #sox ./$file -r 8000 ./$new.wav

  #if using 16khz with low pass at 4khz
  #sox ./$file -r 16000 ./$new.wav lowpass 4000

  dur="$(sox --i -D ./$bare.wav)"


  #Pass file to REAPER
  ~/Documents/PROJECT_FILES/REAPER/build/reaper -i ./$bare.wav -f ./$bare.f0 -p ./$bare.pm -a

  python2.7 ~/Documents/PROJECT_FILES/test_recordings/pm_to_temp_samp_try_adaptivewindow.py -i ./$bare.pm -w ./$bare.wav -d $dur #| tee -a ./test_output_2_6_ARRAY.txt


done
