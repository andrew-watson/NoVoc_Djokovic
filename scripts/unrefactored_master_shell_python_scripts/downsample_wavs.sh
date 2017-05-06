#!/bin/bash

#Needs to be in the same directory as the wav files
#mkdir downsamp_wavs

cd ~/Documents/PROJECT_FILES/Nick_wav/wav/

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
  sox ./$file -r 8000 ~/Documents/PROJECT_FILES/test_recordings/downsamp_wavs/$bare.wav

  #if using 16khz with low pass at 4khz
  #sox ./$file -r 16000 ./$new.wav lowpass 4000

  #dur="$(sox --i -D ./$bare.wav)"

done
