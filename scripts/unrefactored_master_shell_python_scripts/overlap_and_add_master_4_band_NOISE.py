#! /usr/bin/python2.7
#Run from freq_bands folder - WITH NOISE BANDS FOLDER INSIDE
##RUN WITH
#python ~/Documents/PROJECT_FILES/test_recordings/overlap_and_add_master_4_band_NOISE.py -1k ./0_1_GEN/hvd_608 -2k ./1_2_GEN/hvd_608 -3k ./2_3_GEN/hvd_608 -4k ./3_4_GEN/hvd_608 -s 16000 -d 550 > 4_band_plus_NOISE.txt

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
parser.add_argument('--b0_to_1k', '-1k', help="Input of low/full freq model binary file, minus extension")
parser.add_argument('--b1k_to_2k', '-2k', help = "Input of 1 middle frequency band - binary file minus extension")
parser.add_argument('--b2k_to_3k', '-3k', help = "Input of 2 middle frequency band - binary file minus extension")
parser.add_argument('--b3k_to_4k', '-4k', help="Input of hi freq model binary file, minus extension")
parser.add_argument('--samplerate', '-s', help="Sample rate")
parser.add_argument('--dimensionality', '-d', default=550, help="Dimensionality of the incoming mgc files.")
parser.add_argument('--preemphasis', '-p', action='store_true', help="Select if training data had preemphasis filter - this compensates.")
parser.add_argument('--noise', '-n', action='store_true', help="Retrieve the energy components of the top 4 frequency bands when at 16000hz. SELECT True for attached frequency bands, or False for noise from different file")

args = parser.parse_args()

class Overlap_Add():
    def __init__(self):
        self.master = np.array([])
        self.b0_to_1k = np.array([])
        self.b1_to_2k = np.array([])
        self.b2_to_3k = np.array([])
        self.b3_to_4k = np.array([])
        self.samplerate = args.samplerate
        self.dimensionality = float(args.dimensionality)
        self.norm_scale = 1 # Set normalisation scaling factor (1 sets max value as 1, increase to increase max value and decrease overall volume)
        #self.input = self.read_bin_file(str(args.input)+".mgc")


    def read_bin_file(self, in_file, dimens):
        whole = binio.load_binary_file(in_file, dimens) #This value is the dimensionality (resampling size from the previous encoding step)
        return whole

    def empty(self, input1, input2, input3, input4):

        prev_frame = []
        prev_peak = 0

        frames_0k_to_1k = self.read_bin_file(str(input1)+".mgc", self.dimensionality)
        frames_1k_to_2k = self.read_bin_file(str(input2)+".mgc", self.dimensionality)
        frames_2k_to_3k = self.read_bin_file(str(input3)+".mgc", self.dimensionality)
        frames_3k_to_4k = self.read_bin_file(str(input4)+".mgc", self.dimensionality)
        #print "LO LENGTH = ", len(frames_0k_to_1k)
        #print "MID LENGHT = ", len(mid_frames)
        #print "HI LENGTH = ", len(hi_frames)
        #name = "/noisebands_4_5_6_7_8/"+str(input)[2:]+".mgc"
        noisefile = self.read_bin_file("./noise_abs"+str(input1)[10:]+".mgc", 4)
        #noisefile = self.read_bin_file(name, 4)

        #init_noise = np.random.normal(size=1500)
        init_noise = 2*(np.random.rand(1500))-1

        """Currently basing stretch_ratio on the lowest freq model"""
        for (i, x) in enumerate(frames_0k_to_1k):
            """OPEN"""
            print "==============================================="
            new_frame = frames_0k_to_1k[i][:(self.dimensionality-2)]
            pitchmark = frames_0k_to_1k[i][(self.dimensionality-2)]
            stretch_ratio = frames_0k_to_1k[i][(self.dimensionality-1)]

            nb_4_5 = noisefile[i][0]
            nb_5_6 = noisefile[i][1]
            nb_6_7 = noisefile[i][2]
            nb_7_8 = noisefile[i][3]

            #new_hi_frame = hi_frames[i][:(self.dimensionality-2)]
            new_1k_to_2k_frame = frames_1k_to_2k[i][:(self.dimensionality-2)]
            new_2k_to_3k_frame = frames_2k_to_3k[i][:(self.dimensionality-2)]
            new_3k_to_4k_frame = frames_3k_to_4k[i][:(self.dimensionality-2)]


            ##Re-attach zeros to ends ##
            new_frame = np.insert(new_frame, len(new_frame), 0.0)
            new_frame = np.insert(new_frame, 0, 0.0)
            new_1k_to_2k_frame = np.insert(new_1k_to_2k_frame, len(new_1k_to_2k_frame), 0.0)
            new_1k_to_2k_frame = np.insert(new_1k_to_2k_frame, 0, 0.0)
            new_2k_to_3k_frame = np.insert(new_2k_to_3k_frame, len(new_2k_to_3k_frame), 0.0)
            new_2k_to_3k_frame = np.insert(new_2k_to_3k_frame, 0, 0.0)
            new_3k_to_4k_frame = np.insert(new_3k_to_4k_frame, len(new_3k_to_4k_frame), 0.0)
            new_3k_to_4k_frame = np.insert(new_3k_to_4k_frame, 0, 0.0)

            #new_frame = self.load_frame(frames, i)
            timestamp = 0.005 * i
            print "i is - ", i, ", timestamp is - ", timestamp

            """RESAMPLE"""
            ##find original frame size
            ori_size = round(len(new_frame) / stretch_ratio)
            ##apply resampling to original size
            new_frame = scipy.signal.resample(new_frame, ori_size)
            new_1k_to_2k_frame = scipy.signal.resample(new_1k_to_2k_frame, ori_size)
            new_2k_to_3k_frame = scipy.signal.resample(new_2k_to_3k_frame, ori_size)
            new_3k_to_4k_frame = scipy.signal.resample(new_3k_to_4k_frame, ori_size)

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

            #if pitchmark < 1:
            #    noise_frame = noise_frame / 5

            """Hypothesis and add"""
            if i == 0:#initialises with first frame
                #print "first frame"
                self.b0_to_1k = new_frame
                self.b1_to_2k = new_1k_to_2k_frame
                self.b2_to_3k = new_2k_to_3k_frame
                self.b3_to_4k = new_3k_to_4k_frame
                self.all_noise = noise_frame
                #initialise prev_peak for next input frame
                if pitchmark < 1:
                    prev_peak = len(new_frame)/2
                else:
                    prev_peak = np.argmax(new_frame)
                prev_frame = new_frame
                prev_1k_to_2k = new_1k_to_2k_frame
                prev_2k_to_3k = new_2k_to_3k_frame
                prev_3k_to_4k = new_3k_to_4k_frame
                prev_noise = noise_frame
                so_far_0_to_1k = self.b0_to_1k
                so_far_1k_to_2k = self.b1_to_2k
                so_far_2k_to_3k = self.b2_to_3k
                so_far_3k_to_4k = self.b3_to_4k
                #so_far_noise = self.all_noise

            else: #so for each new frame
                hypothesis = self.hypothesis(self.b0_to_1k, prev_peak, this_peak)#RETURNS NUMBER
                #print "Hypothesis = ", hypothesis
                if hypothesis > timestamp + 0.0025:
                    #print "No add"
                    prev_frame = new_frame
                    prev_1k_to_2k = new_1k_to_2k_frame
                    prev_2k_to_3k = new_2k_to_3k_frame
                    prev_3k_to_4k = new_3k_to_4k_frame
                    prev_noise = noise_frame

                elif (timestamp - 0.0025) < hypothesis < (timestamp + 0.0025):
                    #print "hypothesis within range"

                    ##CALCULATE DISTANCES AND POSITIONS##
                    prev_peak_to_end = len(so_far_0_to_1k) - prev_peak
                    begin_to_new_peak = this_peak
                    avg_length = (prev_peak_to_end + begin_to_new_peak) / 2.0
                    new_peak_position = prev_peak + avg_length
                    join_position = new_peak_position - this_peak

                    #### PRINT STATEMENTS FOR RECOVERING FRAME TIME DETAILS FOR COMPARISON ####
                    #print "###  VOICING   =  ", pitchmark
                    #print "Start of frame = ", join_position*(1.0/float(self.samplerate))
                    #print "Peak time = ", new_peak_position*(1.0/float(self.samplerate))
                    #print "End of frame = ", (join_position + len(new_frame)) * (1.0/float(self.samplerate))
                    ### OR JUST ###
                    #join_position = hypothesis - this_peak

                    """Initialise temp frames to add together"""
                    if join_position + len(new_frame) <= len(so_far_0_to_1k):
                        #print "No extension"
                        #FOR LOW#
                        lo_ext_prev_mast = so_far_0_to_1k
                        lo_add_frame = np.append(np.zeros(join_position), new_frame)
                        #For each band#
                        ext_prev_mast_1k_to_2k = so_far_1k_to_2k
                        add_frame_1k_to_2k = np.append(np.zeros(join_position), new_1k_to_2k_frame)

                        ext_prev_mast_2k_to_3k = so_far_2k_to_3k
                        add_frame_2k_to_3k = np.append(np.zeros(join_position), new_2k_to_3k_frame)

                        ext_prev_mast_3k_to_4k = so_far_3k_to_4k
                        add_frame_3k_to_4k = np.append(np.zeros(join_position), new_3k_to_4k_frame)

                        ext_noise = self.all_noise
                        add_noise_frame = np.append(np.zeros(join_position), noise_frame)

                        #print "EXT_PREV_MAST =    ", len(ext_prev_mast)
                        #print "ADD FRAME =     ", len(add_frame)
                    else:
                        #print "Extension ==="
                        #FOR LOW#
                        lo_ext_prev_mast = np.append(so_far_0_to_1k, np.zeros(len(new_frame)-avg_length))
                        lo_add_frame = np.append(np.zeros(join_position), new_frame)
                        #For each band#
                        ext_prev_mast_1k_to_2k = np.append(so_far_1k_to_2k, np.zeros(len(new_frame)-avg_length))
                        add_frame_1k_to_2k = np.append(np.zeros(join_position), new_1k_to_2k_frame)

                        ext_prev_mast_2k_to_3k = np.append(so_far_2k_to_3k, np.zeros(len(new_frame)-avg_length))
                        add_frame_2k_to_3k = np.append(np.zeros(join_position), new_2k_to_3k_frame)

                        ext_prev_mast_3k_to_4k = np.append(so_far_3k_to_4k, np.zeros(len(new_frame)-avg_length))
                        add_frame_3k_to_4k = np.append(np.zeros(join_position), new_3k_to_4k_frame)

                        ext_noise = np.append(self.all_noise, np.zeros(len(new_frame)-avg_length))
                        add_noise_frame = np.append(np.zeros(join_position), noise_frame)

                    self.b0_to_1k = lo_ext_prev_mast + lo_add_frame
                    self.b1_to_2k = ext_prev_mast_1k_to_2k + add_frame_1k_to_2k
                    self.b2_to_3k = ext_prev_mast_2k_to_3k + add_frame_2k_to_3k
                    self.b3_to_4k = ext_prev_mast_3k_to_4k + add_frame_3k_to_4k
                    self.all_noise = ext_noise + add_noise_frame


                    #self.master = ext_prev_mast + add_frame

                    prev_frame = new_frame
                    prev_1k_to_2k = new_1k_to_2k_frame
                    prev_2k_to_3k = new_2k_to_3k_frame
                    prev_3k_to_4k = new_3k_to_4k_frame
                    prev_noise = noise_frame
                    #####
                    prev_peak = new_peak_position
                    #####
                    so_far_0_to_1k = self.b0_to_1k
                    so_far_1k_to_2k = self.b1_to_2k
                    so_far_2k_to_3k = self.b2_to_3k
                    so_far_3k_to_4k = self.b3_to_4k


                else:
                    """assume hyp lies before timestamp - 0.0025"""
                    #do some funky shiz
                    while hypothesis < timestamp - 0.0025:
                        #print "catching up "
                        prev_peak_to_end = len(so_far_0_to_1k) - prev_peak
                        begin_to_new_peak = this_peak
                        avg_length = (prev_peak_to_end + begin_to_new_peak) / 2.0
                        new_peak_position = prev_peak + avg_length
                        join_position = new_peak_position - this_peak
                        ### OR JUST ###
                        #join_position = hypothesis - this_peak
                        if join_position + len(prev_frame) <= len(so_far_0_to_1k):
                            #print "No extension"
                            #FOR LOW#
                            lo_ext_prev_mast = so_far_0_to_1k
                            lo_add_frame = np.append(np.zeros(join_position), prev_frame)

                            #For each band#
                            ext_prev_mast_1k_to_2k = so_far_1k_to_2k
                            add_frame_1k_to_2k = np.append(np.zeros(join_position), prev_1k_to_2k)

                            ext_prev_mast_2k_to_3k = so_far_2k_to_3k
                            add_frame_2k_to_3k = np.append(np.zeros(join_position), prev_2k_to_3k)

                            ext_prev_mast_3k_to_4k = so_far_3k_to_4k
                            add_frame_3k_to_4k = np.append(np.zeros(join_position), prev_3k_to_4k)

                            ext_noise = self.all_noise
                            add_noise_frame = np.append(np.zeros(join_position), prev_noise)
                            #print "EXT_PREV_MAST =    ", len(ext_prev_mast)
                            #print "ADD FRAME =     ", len(add_frame)
                        else:
                            #print "Extension ==="
                            #FOR LOW#
                            lo_ext_prev_mast = np.append(so_far_0_to_1k, np.zeros(len(prev_frame)-avg_length))
                            lo_add_frame = np.append(np.zeros(join_position), prev_frame)

                            #For each band#
                            ext_prev_mast_1k_to_2k = np.append(so_far_1k_to_2k, np.zeros(len(prev_1k_to_2k)-avg_length))
                            add_frame_1k_to_2k = np.append(np.zeros(join_position), prev_1k_to_2k)

                            ext_prev_mast_2k_to_3k = np.append(so_far_2k_to_3k, np.zeros(len(prev_2k_to_3k)-avg_length))
                            add_frame_2k_to_3k = np.append(np.zeros(join_position), prev_2k_to_3k)

                            ext_prev_mast_3k_to_4k = np.append(so_far_3k_to_4k, np.zeros(len(prev_3k_to_4k)-avg_length))
                            add_frame_3k_to_4k = np.append(np.zeros(join_position), prev_3k_to_4k)

                            ext_noise = np.append(self.all_noise, np.zeros(len(prev_frame)-avg_length))
                            add_noise_frame = np.append(np.zeros(join_position), prev_noise)

                        self.b0_to_1k = lo_ext_prev_mast + lo_add_frame
                        self.b1_to_2k = ext_prev_mast_1k_to_2k + add_frame_1k_to_2k
                        self.b2_to_3k = ext_prev_mast_2k_to_3k + add_frame_2k_to_3k
                        self.b3_to_4k = ext_prev_mast_3k_to_4k + add_frame_3k_to_4k
                        self.all_noise = ext_noise + add_noise_frame
                        so_far_0_to_1k = self.b0_to_1k
                        prev_peak = len(self.b0_to_1k) - prev_peak_to_end

                        hypothesis = self.hypothesis(self.b0_to_1k, prev_peak, this_peak)
                        #print "Update     = ", hypothesis
                        #if hypothesis
                    #print "Hypothesis is now True"
                    ##NOW ADD NEW FRAME
                    ##CALCULATE DISTANCES AND POSITIONS##
                    prev_peak_to_end = len(so_far_0_to_1k) - prev_peak
                    begin_to_new_peak = this_peak
                    avg_length = (prev_peak_to_end + begin_to_new_peak) / 2.0
                    new_peak_position = prev_peak + avg_length
                    join_position = new_peak_position - this_peak
                    ### OR JUST ###
                    #join_position = hypothesis - this_peak

                    """Initialise temp frames to add together"""
                    if join_position + len(new_frame) <= len(so_far_0_to_1k):
                        #print "No extension"
                        #FOR LOW#
                        lo_ext_prev_mast = so_far_0_to_1k
                        lo_add_frame = np.append(np.zeros(join_position), new_frame)
                        #For each band#
                        ext_prev_mast_1k_to_2k = so_far_1k_to_2k
                        add_frame_1k_to_2k = np.append(np.zeros(join_position), new_1k_to_2k_frame)

                        ext_prev_mast_2k_to_3k = so_far_2k_to_3k
                        add_frame_2k_to_3k = np.append(np.zeros(join_position), new_2k_to_3k_frame)

                        ext_prev_mast_3k_to_4k = so_far_3k_to_4k
                        add_frame_3k_to_4k = np.append(np.zeros(join_position), new_3k_to_4k_frame)

                        ext_noise = self.all_noise
                        add_noise_frame = np.append(np.zeros(join_position), noise_frame)

                        #print "EXT_PREV_MAST =    ", len(ext_prev_mast)
                        #print "ADD FRAME =     ", len(add_frame)
                    else:
                        #print "Extension ==="
                        #FOR LOW#
                        lo_ext_prev_mast = np.append(so_far_0_to_1k, np.zeros(len(new_frame)-avg_length))
                        lo_add_frame = np.append(np.zeros(join_position), new_frame)
                        #For each band#
                        ext_prev_mast_1k_to_2k = np.append(so_far_1k_to_2k, np.zeros(len(new_frame)-avg_length))
                        add_frame_1k_to_2k = np.append(np.zeros(join_position), new_1k_to_2k_frame)

                        ext_prev_mast_2k_to_3k = np.append(so_far_2k_to_3k, np.zeros(len(new_frame)-avg_length))
                        add_frame_2k_to_3k = np.append(np.zeros(join_position), new_2k_to_3k_frame)

                        ext_prev_mast_3k_to_4k = np.append(so_far_3k_to_4k, np.zeros(len(new_frame)-avg_length))
                        add_frame_3k_to_4k = np.append(np.zeros(join_position), new_3k_to_4k_frame)

                        ext_noise = np.append(self.all_noise, np.zeros(len(new_frame)-avg_length))
                        add_noise_frame = np.append(np.zeros(join_position), noise_frame)

                    self.b0_to_1k = lo_ext_prev_mast + lo_add_frame
                    self.b1_to_2k = ext_prev_mast_1k_to_2k + add_frame_1k_to_2k
                    self.b2_to_3k = ext_prev_mast_2k_to_3k + add_frame_2k_to_3k
                    self.b3_to_4k = ext_prev_mast_3k_to_4k + add_frame_3k_to_4k
                    self.all_noise = ext_noise + add_noise_frame

                    prev_frame = new_frame
                    prev_1k_to_2k = new_1k_to_2k_frame
                    prev_2k_to_3k = new_2k_to_3k_frame
                    prev_3k_to_4k = new_3k_to_4k_frame
                    prev_noise = noise_frame
                    ######
                    prev_peak = new_peak_position
                    #####
                    so_far_0_to_1k = self.b0_to_1k
                    so_far_1k_to_2k = self.b1_to_2k
                    so_far_2k_to_3k = self.b2_to_3k
                    so_far_3k_to_4k = self.b3_to_4k
                    #so_far = self.master

        """APPLY LOW AND HIGHPASS FILTERS TO EACH SIGNAL"""
        ########
        ##### USING ELLIPTICAL FILTER #####
        #Maybe add condition to calculate level from given freq and samplerate
        ## Low - lowpass
        #b, a = scipy.signal.ellip(2, 5, 40, 0.125, btype='lowpass')
        b, a = scipy.signal.butter(2, 0.125, btype='lowpass')
        self.b0_to_1k = scipy.signal.filtfilt(b, a, self.b0_to_1k)

        #b, a = scipy.signal.ellip(2, 5, 40, [0.125, 0.25], btype='bandpass')
        b, a = scipy.signal.butter(2, [0.125, 0.25], btype='bandpass')
        self.b1_to_2k = scipy.signal.filtfilt(b, a, self.b1_to_2k)

        #b, a = scipy.signal.ellip(2, 5, 40, [0.25, 0.375], btype='bandpass')
        b, a = scipy.signal.butter(2, [0.25, 0.375], btype='bandpass')
        self.b2_to_3k = scipy.signal.filtfilt(b, a, self.b2_to_3k)

        #b, a = scipy.signal.ellip(2, 5, 40, [0.375, 0.5], btype='highpass')
        b, a = scipy.signal.butter(2, [0.375, 0.5], btype='bandpass')
        self.b3_to_4k = scipy.signal.filtfilt(b, a, self.b3_to_4k)

        ########

        ##ADD TOGETHER THE FRAMES###
        self.master = self.b0_to_1k + self.b1_to_2k + self.b2_to_3k + self.b3_to_4k

        ##Seperation##
        #b, a = scipy.signal.ellip(2, 5, 40, 0.5, btype='lowpass')
        b, a = scipy.signal.butter(2, 0.5, btype='lowpass')
        master_lo_pass = scipy.signal.filtfilt(b, a, self.master)

        #b, a = scipy.signal.ellip(2, 5, 40, 0.5, btype='highpass')
        b, a = scipy.signal.butter(2, 0.5, btype='highpass')
        noise_hi_pass = scipy.signal.filtfilt(b, a, self.all_noise)

        self.master = self.master + (self.all_noise)
        #self.master = 2*master_lo_pass + (noise_hi_pass/10) #(self.all_noise/5)



        #print "MAX = ", np.max(self.master)
        #self.master = self.master.astype('int32')
        #print "CONVERT MAX = ", np.max(self.master)
        #print "Type of all_master", type(self.master)
        #print len(self.master)
        #print np.max(self.master)
        """APPLY THE INVERSE OF THE PREEMPHASIS FILTER"""
        #WRONG FILTER #self.master = np.append(self.master[0],self.master[1:]+0.95*self.master[:-1])
        if args.preemphasis == True:
            self.master = scipy.signal.lfilter([1, 0], [1, -0.95], self.master)
        #print "Type of all_master", type(self.filt_master)
        return self.master

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
        #print "Max after norm = ", np.max(self.normed)
        #scipy.io.wavfile.write(str(args.high_input)+'_lo_script_test.wav', int(self.samplerate), self.normed)
        new_name =  args.b0_to_1k[9:]
        print new_name
        scipy.io.wavfile.write("."+str(new_name)+'_REF_4_bands_PLUS_NOISE_test.wav', int(self.samplerate), self.normed)
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
    wav_arr = S.empty(args.b0_to_1k, args.b1k_to_2k, args.b2k_to_3k, args.b3k_to_4k)
    S.reconstruct_to_wav(wav_arr)
