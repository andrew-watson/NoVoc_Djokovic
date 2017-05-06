#! /usr/bin/python2.7

#example run with
#python ~/Documents/PROJECT_FILES/test_recordings/overlap_and_add_master_PLUS_NOISE_DRAFT_3.py -i ./ref_data/hvd_608 -s 8000 -d 250 > ola_SEPARATE_noise_GEN_test.txt
#with noiseband folder in folder

import sys
import numpy as np
import scipy
import argparse
import itertools
import os
import scipy.io.wavfile as wf
import scipy.signal
import binary_io

parser = argparse.ArgumentParser(
    description='Overlap and Add')
parser.add_argument('--input', '-i', help="Input binary file, minus extension")
parser.add_argument('--samplerate', '-s', help="Sample rate")
parser.add_argument('--dimensionality', '-d', default=550, help="Dimensionality of the incoming mgc files. Default is 550")
parser.add_argument('--preemphasis', '-p', action='store_true', help="Compensates for preemphasis filter. Set to coeff=0.95")
parser.add_argument('--logstretch', '-l', action='store_true', help="Logstretch selected")
parser.add_argument('--noise', '-n', action='store_true', help="Retrieve the energy components of the top 4 frequency bands when at 16000hz. SELECT True for attached frequency bands, or False for noise from different file")
#parser.add_argument('--difference', '-diff', action='store_true', help="Use this if input data is lower sample rate than intended output samplerate")
#parser.add_argument('--verbose', '-v', help="Print out sanity check print statements")

args = parser.parse_args()

class Overlap_Add():
    def __init__(self):
        self.master = np.array([])
        self.samplerate = args.samplerate
        self.dimensionality = float(args.dimensionality)
        self.norm_scale = 1 # Set normalisation scaling factor (1 sets max value as 1, increase to increase max value and decrease overall volume)
        #self.input = self.read_bin_file(str(args.input)+".mgc")


    def read_bin_file(self, in_file, dimens):
        whole = binio.load_binary_file(in_file, dimens) #This value is the dimensionality (resampling size from the previous encoding step)
        return whole

    def empty(self, input):

        prev_frame = []
        prev_peak = 0
        print "input is  =  ", str(input)

        frames = self.read_bin_file(str(input)+".mgc", self.dimensionality)

        if args.noise == False:
            print "Retrieve noise from other file (other folder)"
            #print "noisebands_4_5_6_7_8/"+str(input)[2:]+".mgc"
            #noisefile = self.read_bin_file("~/Documents/PROJECT_FILES/test_recordings/SECOND_TESTS/noisebands_4_5_6_7_8/"+str(input)[2:]+".mgc", 4)
            noisefile = self.read_bin_file("noise_abs/"+str(input)[2:]+".mgc", 4)

        #init_noise = np.random.normal(size=1000)
        init_noise = 2*(np.random.rand(1500))-1

        print "Reading frames"
        if args.preemphasis == True:
            print "Preemphasis data used - will apply compensation filter"
        if args.logstretch == True:
            print "Log stretch ratio data used - will reverse"
        if args.noise == True:
            print "Retrieving noise bands from file"



        for (i, x) in enumerate(frames):
            """OPEN"""
            #print "==============================================="

            if args.noise == True:
                nb_4_5 = frames[i][(self.dimensionality-5)]
                nb_5_6 = frames[i][(self.dimensionality-4)]
                nb_6_7 = frames[i][(self.dimensionality-3)]
                nb_7_8 = frames[i][(self.dimensionality-2)]

                new_frame = frames[i][:(self.dimensionality-6)]
                pitchmark = frames[i][(self.dimensionality-6)]
                bare_stretch = frames[i][(self.dimensionality-1)]
            else:
                new_frame = frames[i][:(self.dimensionality-2)]
                pitchmark = frames[i][(self.dimensionality-2)]
                bare_stretch = frames[i][(self.dimensionality-1)]

                ## RETRIEVE noise from file ##
                nb_4_5 = noisefile[i][0]
                nb_5_6 = noisefile[i][1]
                nb_6_7 = noisefile[i][2]
                nb_7_8 = noisefile[i][3]


            ##EXPECTS LOG STRETCH_RATIO##
            if args.logstretch == True:
                stretch_ratio = np.exp((bare_stretch)/20)
                print "Recontructed stretch ratio = ", stretch_ratio
            else:
                stretch_ratio = bare_stretch

            ##Re-attach zeros to ends ##
            new_frame = np.insert(new_frame, len(new_frame), 0.0)
            new_frame = np.insert(new_frame, 0, 0.0)

            #new_frame = self.load_frame(frames, i)
            timestamp = 0.005 * i
            #print "i is - ", i, ", timestamp is - ", timestamp

            """RESAMPLE"""
            ##find original frame size
            ori_size = round(len(new_frame) / stretch_ratio)
            ##apply resampling to original size
            new_frame = scipy.signal.resample(new_frame, ori_size)

            """FIND PEAK"""
            if pitchmark < 1:
                this_peak = len(new_frame)/2
            else:
                this_peak = np.argmax(new_frame)



            """Initialise noise frame"""
            #raw_noise = np.random.normal(size=len(new_frame))
            #p = len(new_frame)
            raw_noise = init_noise[:len(new_frame)]

            #b, a = scipy.signal.ellip(2, 5, 40, [0.5, 0.625], btype='bandpass')
            b, a = scipy.signal.butter(2, [0.5, 0.625], btype='bandpass')
            d = scipy.signal.filtfilt(b, a, raw_noise)
            #d = d / np.max(np.abs(d))

            #b, a = scipy.signal.ellip(2, 5, 40, [0.625, 0.75], btype='bandpass')
            b, a = scipy.signal.butter(2, [0.625, 0.75], btype='bandpass')
            e = scipy.signal.filtfilt(b, a, raw_noise)
            #e = e / np.max(np.abs(e))

            #b, a = scipy.signal.ellip(2, 5, 40, [0.75, 0.875], btype='bandpass')
            b, a = scipy.signal.butter(2, [0.75, 0.875], btype='bandpass')
            f = scipy.signal.filtfilt(b, a, raw_noise)
            #f = f / np.max(np.abs(f))

            #b, a = scipy.signal.ellip(2, 5, 40, [0.875, 1], btype='bandpass')
            b, a = scipy.signal.butter(2, [0.875, 1], btype='bandpass')
            g = scipy.signal.filtfilt(b, a, raw_noise)
            #g = g / np.max(np.abs(g))

            noise_frame = (d * nb_4_5) + (e * nb_5_6) + ( f * nb_6_7) + (g * nb_7_8)

            noise_frame = noise_frame * self.create_window(this_peak, len(new_frame))

            #if pitchmark >= 1:
            #    noise_frame = noise_frame / 5



            """Hypothesis and add"""
            if i == 0:#initialises with first frame
                #print "first frame"
                self.master = new_frame
                self.all_noise = noise_frame
                #initialise prev_peak for next input frame
                if pitchmark < 1:
                    prev_peak = len(new_frame)/2
                else:
                    prev_peak = np.argmax(new_frame)
                prev_frame = new_frame
                prev_noise = noise_frame
                so_far = self.master

            else: #so for each new frame
                hypothesis = self.hypothesis(self.master, prev_peak, this_peak)#RETURNS NUMBER
                #print "Hypothesis = ", hypothesis
                if hypothesis > timestamp + 0.0025:
                    #print "No add"
                    prev_frame = new_frame
                    prev_noise = noise_frame
                    ## #YouShallNot
                    #pass
                elif (timestamp - 0.0025) < hypothesis < (timestamp + 0.0025):
                    #print "hypothesis within range"
                    #add this current frame and BE GONE WITH YOU!
                    ##CALCULATE DISTANCES AND POSITIONS##
                    prev_peak_to_end = len(so_far) - prev_peak
                    begin_to_new_peak = this_peak
                    avg_length = (prev_peak_to_end + begin_to_new_peak) / 2.0
                    new_peak_position = prev_peak + avg_length
                    join_position = new_peak_position - this_peak
                    ### OR JUST ###
                    #join_position = hypothesis - this_peak

                    """Initialise temp frames to add together"""
                    if join_position + len(new_frame) <= len(so_far):
                        #print "No extension"
                        ext_prev_mast = so_far
                        add_frame = np.append(np.zeros(join_position), new_frame)
                        ext_noise = self.all_noise
                        add_noise_frame = np.append(np.zeros(join_position), noise_frame)
                        #print "EXT_PREV_MAST =    ", len(ext_prev_mast)
                        #print "ADD FRAME =     ", len(add_frame)
                    else:
                        #print "Extension ==="
                        ext_prev_mast = np.append(so_far, np.zeros(len(new_frame)-avg_length))
                        add_frame = np.append(np.zeros(join_position), new_frame)
                        ext_noise = np.append(self.all_noise, np.zeros(len(new_frame)-avg_length))
                        add_noise_frame = np.append(np.zeros(join_position), noise_frame)

                    self.master = ext_prev_mast + add_frame
                    self.all_noise = ext_noise + add_noise_frame
                    prev_frame = new_frame
                    prev_noise = noise_frame
                    prev_peak = new_peak_position
                    so_far = self.master

                else:
                    """assume hyp lies before timestamp - 0.0025"""
                    #do some funky shiz
                    while hypothesis < timestamp - 0.0025:
                        #print "catching up "
                        prev_peak_to_end = len(so_far) - prev_peak
                        begin_to_new_peak = this_peak
                        avg_length = (prev_peak_to_end + begin_to_new_peak) / 2.0
                        new_peak_position = prev_peak + avg_length
                        join_position = new_peak_position - this_peak
                        ### OR JUST ###
                        #join_position = hypothesis - this_peak
                        if join_position + len(prev_frame) <= len(so_far):
                            #print "No extension"
                            ext_prev_mast = so_far
                            add_frame = np.append(np.zeros(join_position), prev_frame)
                            ext_noise = self.all_noise
                            add_noise_frame = np.append(np.zeros(join_position), prev_noise)
                            #print "EXT_PREV_MAST =    ", len(ext_prev_mast)
                            #print "ADD FRAME =     ", len(add_frame)
                        else:
                            #print "Extension ==="
                            ext_prev_mast = np.append(so_far, np.zeros(len(prev_frame)-avg_length))
                            add_frame = np.append(np.zeros(join_position), prev_frame)
                            ext_noise = np.append(self.all_noise, np.zeros(len(prev_frame)-avg_length))
                            add_noise_frame = np.append(np.zeros(join_position), prev_noise)

                        self.master = ext_prev_mast + add_frame
                        self.all_noise = ext_noise + add_noise_frame
                        so_far = self.master
                        prev_peak = len(self.master) - prev_peak_to_end
                        hypothesis = self.hypothesis(self.master, prev_peak, this_peak)
                        #print "Update     = ", hypothesis
                        #if hypothesis
                    #print "Hypothesis is now True"
                    ##NOW ADD NEW FRAME
                    ##CALCULATE DISTANCES AND POSITIONS##
                    prev_peak_to_end = len(so_far) - prev_peak
                    begin_to_new_peak = this_peak
                    avg_length = (prev_peak_to_end + begin_to_new_peak) / 2.0
                    new_peak_position = prev_peak + avg_length
                    join_position = new_peak_position - this_peak
                    ### OR JUST ###
                    #join_position = hypothesis - this_peak

                    """Initialise temp frames to add together"""
                    if join_position + len(new_frame) <= len(so_far):
                        #print "No extension"
                        ext_prev_mast = so_far
                        add_frame = np.append(np.zeros(join_position), new_frame)
                        ext_noise = self.all_noise
                        add_noise_frame = np.append(np.zeros(join_position), noise_frame)
                        #print "EXT_PREV_MAST =    ", len(ext_prev_mast)
                        #print "ADD FRAME =     ", len(add_frame)
                    else:
                        #print "Extension ==="
                        ext_prev_mast = np.append(so_far, np.zeros(len(new_frame)-avg_length))
                        add_frame = np.append(np.zeros(join_position), new_frame)
                        ext_noise = np.append(self.all_noise, np.zeros(len(new_frame)-avg_length))
                        add_noise_frame = np.append(np.zeros(join_position), noise_frame)

                    self.master = ext_prev_mast + add_frame
                    self.all_noise = ext_noise + add_noise_frame
                    prev_frame = new_frame
                    prev_noise = noise_frame
                    prev_peak = new_peak_position
                    so_far = self.master

        if args.preemphasis == True:
            self.master = scipy.signal.lfilter([1, 0], [1, -0.95], self.master)
            #apply inverse filter / compensation

        ##Seperation##

        #b, a = scipy.signal.ellip(2, 5, 40, 0.5, btype='lowpass')
        b, a = scipy.signal.butter(2, 0.5, btype='lowpass')
        master_lo_pass = scipy.signal.filtfilt(b, a, self.master)

        #b, a = scipy.signal.ellip(2, 5, 40, 0.5, btype='highpass')
        b, a = scipy.signal.butter(2, 0.5, btype='highpass')
        noise_hi_pass = scipy.signal.filtfilt(b, a, self.all_noise)

        self.master = master_lo_pass + (noise_hi_pass) #(self.all_noise/5)
        #if args.difference == True:
        #    self.master = scipy.signal.resample(self.master, len(self.master))
        return self.master
        #return self.all_noise

    def hypothesis(self, total_so_far, previous_peak, potential_new_peak):
        ##Just some numbers
        prev_peak_to_end = len(total_so_far) - previous_peak
        #print "PREVPEAK TO END = ", prev_peak_to_end
        begin_to_new_peak = potential_new_peak
        #print "BEGIN TO NEW PEAK = ", begin_to_new_peak
        avg_length = (prev_peak_to_end + begin_to_new_peak) / 2.0
        #print "AVG LENGTH = ", avg_length
        new_peak_position = previous_peak + avg_length
        #print "NEW PEAK POS = ", new_peak_position
        time = (1.0 / float(self.samplerate)) * new_peak_position
        #print "TIME IS = ", time
        return time


    def reconstruct_to_wav(self, arr):
        self.normed = self.normalise(self.master)
        #scipy.io.wavfile.write(str(args.input)+'_logstretch_copysynth.wav', int(self.samplerate), self.normed)
        #scipy.io.wavfile.write(str(args.input)+'_8khz_noise_16000.wav', 16000, self.normed)
        if args.preemphasis == True:
            scipy.io.wavfile.write(str(args.input)+'_noise_preemph_16000.wav', 16000, self.normed)
        else:
            scipy.io.wavfile.write(str(args.input)+'_noise_16000.wav', 16000, self.normed)
        print "Writing to wav..."

    def normalise(self, array):
        max_val = self.norm_scale * np.max(np.absolute(array))
        new_array = array/max_val
        #print max_val
        return new_array

    def create_window(self, peak, length):

        pos = peak
        neg_pos = length - peak
        first=np.hanning(2*pos)
        second=np.hanning(2*neg_pos)

        new_window = np.append((first[:pos]), (second[neg_pos:]))
        return new_window








if __name__ == "__main__":
    binio = binary_io.BinaryIOCollection()
    S = Overlap_Add()
    wav_arr = S.empty(args.input)
    S.reconstruct_to_wav(wav_arr)
