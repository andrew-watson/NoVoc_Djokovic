#! /usr/bin/python2.7
#PASS PITCHMARK FILE and corresponding wav file TO THIS FUNCTION AND DURATION(from the parent shell script)
#eg run with
#python ./pm_to_samp_adaptivewindow_framing_PLUS_PM.py -i ./herald_001.pm -w ./herald_001.wav -d 1.64


import sys
import numpy
import scipy
import argparse
import itertools
import os
import subprocess
import scipy.io.wavfile as wf
import scipy.signal


parser = argparse.ArgumentParser(
    description='Framing, resampling, windowing.')
parser.add_argument('--input', '-i', help="Pitchmark file")
parser.add_argument('--wav_reference', '-w', help="Reference Wav file")
parser.add_argument('--duration', '-d', help="Duration of wav file")
parser.add_argument('--preemphasis', '-p', action='store_true', help="Add a preemphasis filter. Set to coeff=0.95")
parser.add_argument('--logstretch', '-l', action='store_true', help='Encodes the scaled log of the stretch factor - (multiplied by 20)')
parser.add_argument('--filter', '-f', nargs='*', help='Select a filter - lowpass, highpass, bandpass and values')
parser.add_argument('--samplerate', '-s', default=16000)

args = parser.parse_args()
#####################################################

class Proc():
    def __init__(self):
        self.tracker=[]
        self.samplerate = args.samplerate #Default Sample Rate 16000hz
        self.wav_ref = args.wav_reference
        self.resamplesize = 550 #SET RESAMPLING SIZE
        self.timestamp_list = []

    def whole_function(self, input_file):
        data = self.read_from_file(input_file)
        self.get_time_ranges(data)

    def read_from_file(self, filename):
        
        """Read pitchmark file and populate internal pitchmark tracker"""

        self.filename = filename
        self.bare = filename[4:-3] #Strip file extension

        with open(filename) as f:
            for line in f:
                current = line.split()
                try:
                    current = [float(x) for x in current]
                    self.tracker.append(current)
                    self.timestamp_list.append(current[0])
                except:
                    pass
        return self.tracker



    def get_time_ranges(self, tracker):

        master=[]
        timestamp_list=[]
        new_tracker = []

        self.read(self.wav_ref)

        for value in tracker:
            new_tracker.append(value[0])

        #Corresponding front-end linguistic features are extracted at 5ms intervals,
        #so there must be a 1:1 mapping between linguistic and acoustic frames for training
            
        for j in numpy.arange(0.000, float(args.duration), 0.005):

            #Find nearest pitchmark timestamp for each 5ms timestep#
            time = min(self.timestamp_list, key=lambda x:abs(x-j))
            #Find corresponding index of pitchmark#
            ar = numpy.where(numpy.array(new_tracker) == time)
            i = int(ar[0])

            #Pitchmark index from original reference wav file
            self.current = self.tracker[i][0]

            #Extract time boundaries (ms) of current frame#
            #Repeated condition if a different fixed frame shift (instead of REAPER's 10ms) is to be used#

            #If no pitch mark
            if self.tracker[i][1] == 0:
                if i == 0:
                    #Beginning of the whole audio track#
                    self.beginning = 0.00
                else:
                    self.beginning = self.tracker[i-1][0]
            #If pitch mark present        
            else:
                if i == 0:
                    self.beginning = 0.00
                else:
                    self.beginning = self.tracker[i-1][0]

            #End of the audio track (REAPER's output stops before the end/duration of full audio track)#
            if i == len(self.tracker)-1:
                self.end = float(args.duration)
            else:
                self.end = self.tracker[i+1][0]

            #Retrieve indeces of frame beginning, current peak, and frame end##
            start_samp = round(self.beginning*self.samplerate)
            current_samp = round(self.current*self.samplerate)
            end_samp = round(self.end*self.samplerate)

            #Retrieve frame from audio track#
            frame = self.wave[int(start_samp):int(end_samp)]

            ##Calculating the stretch ratio#
            frame_length = len(frame)
            stretch_ratio = self.resamplesize / float(frame_length)

            #Upsampling the frame to fixed frame size#
            resamp = scipy.signal.resample(frame, self.resamplesize) #Second argument can change based on max size.

            #Create and apply Hanning Window#
            hann = self.create_window(start_samp, current_samp, end_samp, self.resamplesize)

            if len(hann) != len(resamp):
                #for any mismatches or rounding error#
                window = resamp * hann[:-1]
            else:
                window = resamp * hann

            #Preparing frame for encoding#
            new_win = window[1:-1] #First and last are always 0, which cannot be normalised for NN training. They are added back in/accounted for at the overlap and add step.

            #Appends the pitchmark flag as a feature#
            new_win = numpy.append(new_win, self.tracker[i][1])

            #Appends the stretch ratio as a feature#
            if args.logstretch == True:
                new_win = numpy.append(new_win, 20*numpy.log(stretch_ratio)) ##Appends the log of the stretch ratio * 20.
            else:
                new_win = numpy.append(new_win, stretch_ratio) ##Appends the bare stretch ratio

            #Add current frame of features to total array#
            master.append(new_win)

        """Save to binary output file"""
        print "Saving ......................"
        name = self.naming_convention(self.bare)

        #To txt file for debugging#
        numpy.savetxt("./mgc/"+str(name)+".txt", master)
        #To binary file - mgc - for training DNN#
        self.array_to_binary_file(master, "./mgc/"+str(name)+".mgc")
        print "================================================================"



    def naming_convention(self, name):
        """Fancy naming convention to keep track of data file content"""
        if args.preemphasis == True:
            p_morph = "_preemph"
        else:
            p_morph = ""
        if args.logstretch == True:
            l_morph = "_logstretch"
        else:
            l_morph = ""
        if args.filter == True:
            if args.filter[0] == 'bandpass':
                f_morph = "_"+str(args.filter[1])+"_"+str(args.filter[2])
            elif args.filter[0] == 'lowpass':
                f_morph = "_"+str(args.filter[1])+"_lowpass"
            elif args.filter[0] == 'highpass':
                f_morph = "_"+str(args.filter[1])+"_highpass"
            else:
                print "oops"
        else:
            f_morph = ""

        concat_name = str(name)+str(f_morph)+str(p_morph)+str(l_morph)
        return concat_name



    def extract_segment(self, filename, start, end):
        subprocess.call(["sox", str(filename[:-3]+".wav"), "./temp.wav", "trim", str(start), str(end)])



    def extract_duration(self, filename):
        subprocess.call(["sox", --i, -D, str(filename)])



    def read(self, path):

        """Read wav file and apply necessary filters"""

        data = wf.read(path)

        if args.filter == True:
            print "filter is true"
            if args.filter[0] == 'bandpass':
                print "Bandpass selected = ", args.filter[1], " - ", args.filter[2]
                top = float(args.filter[2])/self.samplerate
                bottom = float(args.filter[1])/self.samplerate
                b, a = scipy.signal.ellip(2, 5, 40, [bottom, top], btype='bandpass')
                self.wave = scipy.signal.filtfilt(b, a, data[1])
            elif args.filter[0] == 'lowpass':
                print "Lowpass selected - ", args.filter[1]
                b, a = scipy.signal.ellip(2, 5, 40, float(args.filter[1])/self.samplerate, btype='lowpass')
                self.wave = scipy.signal.filtfilt(b, a, data[1])
            elif args.filter[0] == 'highpass':
                print "Highpass selected", args.filter[1]
                b, a = scipy.signal.ellip(2, 5, 40, float(args.filter[1])/self.samplerate, btype='highpass')
                self.wave = scipy.signal.filtfilt(b, a, data[1])
            else:
                "#NoFilter"
                self.wave = data[1]

        if args.preemphasis == True:
            print "Preemphasis selected"
            self.wave = scipy.signal.lfilter([1, -0.95], 1, data[1])
        else:
            print "No preemphasis"
            self.wave = data[1]



    def create_window(self, start, peak, end, resampled):

        """Creates an asymmetric Hanning window based on the current frame size and peak position"""

        pos = round(((peak - start)/(end-start)) * resampled)
        neg_pos = round(((end - peak) / (end-start)) * resampled)

        first=numpy.hanning(2*pos)
        second=numpy.hanning(2*neg_pos)

        new_window = numpy.append((first[:int(pos)]), (second[int(neg_pos):]))

        return new_window


#######################################################################################
########### Taken from binary_io.py from the DNN toolkit ##############################
    def array_to_binary_file(self, data, output_file_name):
        data = numpy.array(data, 'float32')

        fid = open(output_file_name, 'wb')
        data.tofile(fid)
        fid.close()

#######################################################################################

if __name__ == "__main__":
    S = Proc()
    S.whole_function(args.input)
