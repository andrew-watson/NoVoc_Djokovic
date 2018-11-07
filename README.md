# NoVoc Djokovic - raw waveform 

## What?

These are the baseline scripts developed for my Master's dissertation on the MSc Speech & Language Processing course at the University of Edinburgh (2015-2016).

My project researched and developed methods of modelling raw waveforms in a deep neural network speech synthesis system, instead of predicting vocoder parameters as in other systems...hence the (terrible) name, NoVoc Djokovic.

In this repository are the scripts for creating the training data to pass to a neural speech synthesis system, and the scripts for concatenating the resultant waveforms. As they stand, the pipeline can be run end-to-end as a proof of concept. This will split the original audio file, and reconstruct it as it would after synthesis time.

For this project, I used CSTR's Neural Network based Speech Synthesis System (https://github.com/CSTR-Edinburgh/merlin/).

## How to run

First, you will need the following command line tools:

* REAPER: Robust Epoch And Pitch EstimatoR - https://github.com/google/REAPER

* SoX - Sound eXchange - http://sox.sourceforge.net/

The scripts should be run from the top directory, NoVoc_Djokovic

```
cd NoVoc_Djokovic

./scripts/create_data/process_training_data.sh
```

This runs the following steps:
* Downsamples your audio files to 16kHz, using SoX
* Extracts pitchmarks from each downsampled audio file, using REAPER
* Uses the pitchmark information to extract waveform segments from each audio file. Each segment is a window centred either on each pitchmark (for voiced segments of speech), or using a constant frame size (for unvoiced segments of speech).
* Resamples each segment so they are frames of equal length to pass to the neural network. A stretch factor is also appended as a feature to be able to resample to the correct size at synthesis time.
* Saves a (binary) .mgc file for each audio file, containing all extracted frames. .mgc is used as this is an extension Merlin expects for training data.


To produce the resultant audio file from the generated waveform frames at synthesis time, the following script performs an overlap and add to concatenate the frames together into speech.

`./scripts/synthesis_time/baseline_OLA.sh`

The above command will currently take the .mgc produced by the first data preparation step to re-concatenate into an audio file of speech. In practice, this script will take the generated .mgc file, but for now it points to the original training data so you can try it out.

