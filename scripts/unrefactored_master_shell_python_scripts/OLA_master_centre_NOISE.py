#! /usr/bin/python2.7

#run with
#python ~/Documents/PROJECT_FILES/test_recordings/OLA_master_centre_NOISE.py -i ./GEN/hvd_609 -s 16000 -d 959 > ola_SEPARATE_noise_GEN_test.txt
#script always assumes that the noiseband folder is in the working directory.
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
parser.add_argument('--dimensionality', '-d', default=550, help="Dimensionality of the incoming mgc files.")
parser.add_argument('--preemphasis', '-p', action='store_true', help="Compensates for preemphasis filter. Set to coeff=0.95")
parser.add_argument('--join_method', '-j', default='absolute', help="Choose overlap join method. Default is absolute (peak to peak), or use average")
parser.add_argument('--threshold', '-t', default=0, help="Choose threshold level for finding the beginning and end of the frame. Default is 0.")
parser.add_argument('--noise', '-n', action='store_true', help="Retrieve the energy components of the top 4 frequency bands when at 16000hz. SELECT True for attached frequency bands, or False for noise from different file")

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

        frames = self.read_bin_file(str(input)+".mgc", self.dimensionality)

        print "Reading frames"
        if args.preemphasis == True:
            print "Preemphasis data used - will apply compensation filter"
        if args.noise == True:
            print "Retrieving noise bands from file"

        noisefile = self.read_bin_file("noise_abs/"+str(input)[2:]+".mgc", 4)

        #init_noise = np.random.normal(size=1500)
        init_noise = 2*(np.random.rand(1500))-1


        #print "max init noise = ", np.max(np.abs(init_noise))

        print "Overlap method is = ", str(args.join_method)

        for (i, x) in enumerate(frames):
            """OPEN"""
            print "==============================================="
            new_frame = frames[i][:(self.dimensionality-1)]
            pitchmark = frames[i][(self.dimensionality-1)]

            nb_4_5 = noisefile[i][0]
            nb_5_6 = noisefile[i][1]
            nb_6_7 = noisefile[i][2]
            nb_7_8 = noisefile[i][3]

            ##Re-attach zeros to ends ##
            #new_frame = np.insert(new_frame, len(new_frame), 0.0)
            #new_frame = np.insert(new_frame, 0, 0.0)

            #new_frame = self.load_frame(frames, i)
            timestamp = 0.005 * i
            print "i is - ", i, ", timestamp is - ", timestamp
            print "Pitchmark =======  ", pitchmark

            """FIND ORIGINAL SIZE"""
            trim_b = np.min(np.where(abs(new_frame) > float(args.threshold)))
            trim_e = (np.max(np.where(abs(new_frame) > float(args.threshold))) + 1)

            new_frame = new_frame[trim_b:trim_e]
            ##find original frame size
            #ori_size = round(len(new_frame) / stretch_ratio)
            ##apply resampling to original size
            #new_frame = scipy.signal.resample(new_frame, ori_size)

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
            #print "max d = ", np.max(np.abs(d))

            #b, a = scipy.signal.ellip(2, 5, 40, [0.625, 0.75], btype='bandpass')
            b, a = scipy.signal.butter(2, [0.625, 0.75], btype='bandpass')
            e = scipy.signal.filtfilt(b, a, raw_noise)
            #print "max e = ", np.max(np.abs(e))
            #e = e / np.max(np.abs(e))

            #b, a = scipy.signal.ellip(2, 5, 40, [0.75, 0.875], btype='bandpass')
            b, a = scipy.signal.butter(2, [0.75, 0.875], btype='bandpass')
            f = scipy.signal.filtfilt(b, a, raw_noise)
            #print "max f= ", np.max(np.abs(f))
            #f = f / np.max(np.abs(f))

            #b, a = scipy.signal.ellip(2, 5, 40, [0.875, 1], btype='bandpass')
            b, a = scipy.signal.butter(2, [0.875, 1], btype='bandpass')
            g = scipy.signal.filtfilt(b, a, raw_noise)
            #print "max g = ", np.max(np.abs(g))
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
                    #print "Current length is = ", len(self.master)
                    #print "Time is = ", len(self.master)*(1.0 / float(self.samplerate))
                    ## #YouShallNot
                    #pass
                elif (timestamp - 0.0025) < hypothesis < (timestamp + 0.0025):
                    #print "hypothesis within range"
                    #add this current frame and BE GONE WITH YOU!
                    ##CALCULATE DISTANCES AND POSITIONS##
                    prev_peak_to_end = len(so_far) - prev_peak #DISTANCE
                    #print "prev_peak_to_end = ", prev_peak_to_end
                    begin_to_new_peak = this_peak +1 #DISTANCE
                    #print "begin_to_new_peak = ", begin_to_new_peak
                    """################ EDIT ##################"""
                    if args.join_method == 'average':
                        avg_length = (prev_peak_to_end + begin_to_new_peak) / 2.0
                        #print "avg_length = ", avg_length
                        new_peak_position = prev_peak + avg_length #POSITION
                        #print "new_peak_position = ", new_peak_position
                        join_position = new_peak_position - this_peak
                        #print "join_position = ", join_position
                        neg_len = avg_length


                    elif args.join_method == 'absolute':
                        new_peak_position = prev_peak + this_peak
                        #print "new_peak_position = ", new_peak_position
                        join_position = prev_peak
                        #print "join_position = ", join_position
                        neg_len = prev_peak_to_end


                        #absolute chosen - this is the default
                    else:
                        #print "problem with join/overlap selection"

                    ##Just some checks##
                    #print "Neg_len = ", neg_len


                    ### OR JUST ###
                    #join_position = hypothesis - this_peak
                    """################ EDIT ##################"""

                    """Initialise temp frames to add together"""
                    if join_position + len(new_frame) <= len(so_far):
                        #print "No extension"
                        ext_prev_mast = so_far
                        add_frame = np.append(np.zeros(join_position), new_frame)
                        add_frame = np.append(add_frame, np.zeros(len(so_far)-(join_position+len(new_frame))))
                        ext_noise = self.all_noise
                        add_noise_frame = np.append(np.zeros(join_position), noise_frame)
                        #print "EXT_PREV_MAST =    ", len(ext_prev_mast)
                        #print "ADD FRAME =     ", len(add_frame)
                    else:
                        #print "Extension ==="
                        ext_prev_mast = np.append(so_far, np.zeros(len(new_frame)-neg_len))
                        add_frame = np.append(np.zeros(join_position), new_frame)
                        ext_noise = np.append(self.all_noise, np.zeros(len(new_frame)-neg_len))
                        add_noise_frame = np.append(np.zeros(join_position), noise_frame)
                        #print "EXT_PREV_MAST =    ", len(ext_prev_mast)
                        #print "ADD FRAME =     ", len(add_frame)

                    if len(ext_prev_mast) == len(add_frame):
                        self.master = ext_prev_mast + add_frame
                    else:
                        self.master = ext_prev_mast + add_frame[:-1]
                    self.all_noise = ext_noise + add_noise_frame
                    #print "Current length is = ", len(self.master)
                    #print "Time is = ", len(self.master)*(1.0 / float(self.samplerate))
                    prev_frame = new_frame
                    prev_noise = noise_frame
                    prev_peak = new_peak_position
                    so_far = self.master

                else:
                    """assume hyp lies before timestamp - 0.0025"""
                    #do some funky shiz
                    while hypothesis < timestamp - 0.0025:
                        #print " ====================== catching up ============== "
                        prev_peak_to_end = len(so_far) - prev_peak
                        begin_to_new_peak = this_peak + 1
                        if args.join_method == 'average':
                            #print "=Using average"
                            #print "prev peak is = ", prev_peak
                            #print "this peak is = ", this_peak
                            avg_length = (prev_peak_to_end + begin_to_new_peak) / 2.0
                            #print "avg_length = ", avg_length
                            new_peak_position = prev_peak + avg_length
                            #print "New peak position = ", new_peak_position
                            join_position = new_peak_position - this_peak
                            #print "join position = ", join_position
                            neg_len = avg_length
                            #print "neg len = ", neg_len

                        elif args.join_method == 'absolute':
                            #print "=Using absolute"
                            #print "prev peak is = ", prev_peak
                            #print "this peak is = ", this_peak
                            new_peak_position = prev_peak + this_peak
                            #print "New peak position = ", new_peak_position
                            join_position = prev_peak
                            #print "join position = ", join_position
                            neg_len = prev_peak_to_end
                            #print "neg len = ", neg_len
                            #absolute chosen - this is the default
                        else:
                            #print "problem with join/overlap selection"
                        ### OR JUST ###
                        #join_position = hypothesis - this_peak
                        """Initialise temp frames to add together"""
                        if join_position + len(prev_frame) <= len(so_far):
                            #print "No extension"
                            ext_prev_mast = so_far
                            add_frame = np.append(np.zeros(join_position), prev_frame)
                            add_frame = np.append(add_frame, np.zeros(len(so_far)-(join_position+len(prev_frame))))
                            ext_noise = self.all_noise
                            add_noise_frame = np.append(np.zeros(join_position), prev_noise)

                            #print "EXT_PREV_MAST =    ", len(ext_prev_mast)
                            #print "ADD FRAME =     ", len(add_frame)
                        else:
                            #print "==Extension ==="
                            ext_prev_mast = np.append(so_far, np.zeros(len(prev_frame)-neg_len))
                            add_frame = np.append(np.zeros(join_position), prev_frame)
                            ext_noise = np.append(self.all_noise, np.zeros(len(prev_frame)-neg_len))
                            add_noise_frame = np.append(np.zeros(join_position), prev_noise)
                            #print "EXT_PREV_MAST =    ", len(ext_prev_mast)
                            #print "ADD FRAME =     ", len(add_frame)
                        if len(ext_prev_mast) == len(add_frame):
                            self.master = ext_prev_mast + add_frame
                        else:
                            self.master = ext_prev_mast + add_frame[:-1]
                        self.all_noise = ext_noise + add_noise_frame
                        so_far = self.master
                        prev_peak = len(self.master) - prev_peak_to_end
                        #print "Some values = ", self.master[-10:]

                        #print "Current length is = ", len(self.master)
                        #print "Time is = ", len(self.master)*(1.0 / float(self.samplerate))
                        hypothesis = self.hypothesis(self.master, prev_peak, this_peak)
                        #print "Update     = ", hypothesis
                        #if hypothesis
                    #print "Hypothesis is now True"
                    ##NOW ADD NEW FRAME
                    ##CALCULATE DISTANCES AND POSITIONS##
                    prev_peak_to_end = len(so_far) - prev_peak
                    begin_to_new_peak = this_peak + 1
                    if args.join_method == 'average':
                        avg_length = (prev_peak_to_end + begin_to_new_peak) / 2.0
                        new_peak_position = prev_peak + avg_length
                        join_position = new_peak_position - this_peak
                        neg_len = avg_length

                    elif args.join_method == 'absolute':
                        new_peak_position = prev_peak + this_peak
                        join_position = prev_peak
                        neg_len = prev_peak_to_end
                        #absolute chosen - this is the default
                    else:
                        #print "problem with join/overlap selection"

                    ### OR JUST ###
                    #join_position = hypothesis - this_peak

                    """Initialise temp frames to add together"""
                    if join_position + len(new_frame) <= len(so_far):
                        #print "No extension"
                        ext_prev_mast = so_far
                        add_frame = np.append(np.zeros(join_position), new_frame)
                        add_frame = np.append(add_frame, np.zeros(len(so_far)-(join_position+len(new_frame))))
                        ext_noise = self.all_noise
                        add_noise_frame = np.append(np.zeros(join_position), noise_frame)
                        #print "EXT_PREV_MAST =    ", len(ext_prev_mast)
                        #print "ADD FRAME =     ", len(add_frame)
                    else:
                        #print "Extension ==="
                        ext_prev_mast = np.append(so_far, np.zeros(len(new_frame)- neg_len))
                        add_frame = np.append(np.zeros(join_position), new_frame)
                        ext_noise = np.append(self.all_noise, np.zeros(len(new_frame)-neg_len))
                        add_noise_frame = np.append(np.zeros(join_position), noise_frame)
                        #print "EXT_PREV_MAST =    ", len(ext_prev_mast)
                        #print "ADD FRAME =     ", len(add_frame)


                    """UNACCEPTABLE CONDITIOOOONSSSAH""" #probably due to rounding error.
                    if len(ext_prev_mast) == len(add_frame):
                        self.master = ext_prev_mast + add_frame
                    else:
                        self.master = ext_prev_mast + add_frame[:-1]
                    self.all_noise = ext_noise + add_noise_frame
                    #self.master = ext_prev_mast + add_frame



                    print "Current length is = ", len(self.master)
                    #print "Time is = ", len(self.master)*(1.0 / float(self.samplerate))
                    prev_frame = new_frame
                    prev_noise = noise_frame
                    prev_peak = new_peak_position
                    so_far = self.master

        #if args.preemphasis == True:
            #self.master = apply inverse filter / compensation

        ##Seperation##
        #b, a = scipy.signal.ellip(2, 5, 40, 0.5, btype='lowpass')
        b, a = scipy.signal.butter(2, 0.5, btype='lowpass')
        master_lo_pass = scipy.signal.filtfilt(b, a, self.master)

        #b, a = scipy.signal.ellip(2, 5, 40, 0.5, btype='highpass')
        b, a = scipy.signal.butter(2, 0.5, btype='highpass')
        noise_hi_pass = scipy.signal.filtfilt(b, a, self.all_noise)

        self.master = master_lo_pass + noise_hi_pass #(self.all_noise/5)

        return self.master

    def hypothesis(self, total_so_far, previous_peak, potential_new_peak):
        prev_peak_to_end = len(total_so_far) - previous_peak
        #print "PREVPEAK TO END = ", prev_peak_to_end
        begin_to_new_peak = potential_new_peak + 1
        if args.join_method == 'average':
            avg_length = (prev_peak_to_end + begin_to_new_peak) / 2.0
            new_peak_position = previous_peak + avg_length
            #join_position = new_peak_position - this_peak
        elif args.join_method == 'absolute':
            new_peak_position = previous_peak + potential_new_peak
            #join_position = previous_peak
            #absolute chosen - this is the default
        else:
            print "problem with join/overlap selection"
        #print "======HYPOETHESIS FINCTION"
        #print "NEW PEAK POS = ", new_peak_position
        time = (1.0 / float(self.samplerate)) * new_peak_position
        #print "TIME IS = ", time
        #print "=========================="
        return time


    def reconstruct_to_wav(self, arr):
        self.normed = self.normalise(self.master)
        #scipy.io.wavfile.write(str(args.input)+'_logstretch_copysynth.wav', int(self.samplerate), self.normed)
        scipy.io.wavfile.write(str(args.input)+'_'+str(args.join_method)+'_'+str(args.threshold)+'_CENTRE_noise.wav', int(self.samplerate), self.normed)
        print "Writing to wav..."

    def normalise(self, array):
        max_val = self.norm_scale * np.max(np.absolute(array))
        new_array = array/max_val
        print max_val
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
