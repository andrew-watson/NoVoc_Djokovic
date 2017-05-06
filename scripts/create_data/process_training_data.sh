#!/bin/bash

#Run from NoVoc_Djokovic


#Downsampling to 16000

echo "Downsampling wavs to 16000"

if [ ! -d ./downsampled_wavs ]; then
	mkdir ./downsampled_wavs
fi

cd ./wav

for file in ./*.wav ; do
  bare=${file:2:-4}
  echo $bare
  sox ./$file -r 16000 ../downsampled_wavs/$bare.wav
done

#Create pitchmark files

echo "Creating pitchmark files with REAPER"

cd ..

for file in ./downsampled_wavs/*.wav ; do
  #Pass file to REAPER
  bare=${file:19:-4}
  echo $bare
  ~/Documents/REAPER/build/reaper -i $file -p ./pm/$bare.pm -a
done

#Create mgcs for every downsampled wav with pm file. Also takes duration as input

echo "Creating mgcs"

for file in ./downsampled_wavs/*.wav ; do
  bare=${file:19:-4}
  echo $bare
  dur="$(sox --i -D $file)"

  python ./scripts/create_data/refactor_write_to_mgcs_master.py -i ./pm/$bare.pm -w $file -d $dur
done

echo "Done"

