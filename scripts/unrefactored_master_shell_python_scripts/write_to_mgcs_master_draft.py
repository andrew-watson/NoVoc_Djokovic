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
#import binary_io


parser = argparse.ArgumentParser(
    description='Framing, resampling, windowing.')
parser.add_argument('--input', '-i', help="Pitchmark file")
parser.add_argument('--wav_reference', '-w', help="Reference Wav file")
parser.add_argument('--duration', '-d', help="Duration of wav file")
parser.add_argument('--preemphasis', '-p', action='store_true', help="Add a preemphasis filter. Set to coeff=0.95")
parser.add_argument('--logstretch', '-l', action='store_true', help='Encodes the scaled log of the stretch factor - (multiplied by 20)')
parser.add_argument('--filter', '-f', nargs='*', help='Select a filter - lowpass, highpass, bandpass and values')

args = parser.parse_args()
#####################################################

class Proc():
    def __init__(self):
        self.tracker=[]
        self.samplerate = 16000 #SET SAMPLERATE
        self.wav_ref = args.wav_reference
        self.resamplesize = 550 #SET RESAMPLING SIZE
        self.timestamp_list = []

    def whole_function(self, input_file):
        data = self.read_from_file(input_file)
        self.get_time_ranges(data)

    def read_from_file(self, filename):
        self.filename = filename
        self.bare = filename[:-3]

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


    def myround(self, x, prec=2, base=.05): #Unused
        return round(base * round(float(x)/base),prec)

    def get_time_ranges(self, tracker):

        b = 0
        self.read(self.wav_ref)
        master=[]
        timestamp_list=[]
        new_tracker = []
        for value in tracker:
            new_tracker.append(value[0])

        if args.logstretch == True:
            print "Logstretch selected"

        for j in numpy.arange(0.000, float(args.duration), 0.005):

            #Find nearest timestamp for each timestep
            time = min(self.timestamp_list, key=lambda x:abs(x-j))
            ar = numpy.where(numpy.array(new_tracker) == time)
            i = int(ar[0])

            """CHECKS"""
            #print "j is === ", j
            #print "NEAREST TIMESTAMP IS === ", time
            #print "AS AN INDEX IS ===  ", ar
            #print "i is ====   ", i

            b+=1

            #ORIGINAL PITCHMARK SAMPLE INDEXING#
            self.current = self.tracker[i][0]
            if self.tracker[i][1] == 0: #IF NO PITCHMARK
                if i == 0:
                    self.beginning = 0.00
                else:
                    self.beginning = self.tracker[i-1][0]
            else: #PITCHMARK PRESENT
                if i == 0:
                    self.beginning = 0.00
                else:
                    self.beginning = self.tracker[i-1][0]
            if i == len(self.tracker)-1:
                self.end = float(args.duration)
            else:
                self.end = self.tracker[i+1][0]

            #print "=====TESTS===="
            #print self.beginning,
            #print "---",
            #print self.end
            #print "="

            ## THIS IS THE BULK OF THE WORK ##
            #round(time*samplerate)
            start_samp = round(self.beginning*self.samplerate) #PLUS/MINUS 1?? INDEXING AN ARRAY - but already index 0 at time 0...
            current_samp = round(self.current*self.samplerate)
            end_samp = round(self.end*self.samplerate)

            frame = self.wave[start_samp:end_samp]

            ##Storing the stretch ratio
            frame_length = len(frame)
            #print "FRAME LENGTH IS = ", (frame_length * (1.0 / self.samplerate))
            stretch_ratio = self.resamplesize / float(frame_length)

            #print "FRAME LEGNTH   ", frame_length
            #print "STRETCH   ", stretch_ratio
            #print "RESAMP SIZE", self.resamplesize
            #print len(self.wave)

            resamp = scipy.signal.resample(frame, self.resamplesize) #Second argument can change based on max size.

            """APPLYING WINDOW"""
            ##If static hanning window
            #window = resamp * numpy.hanning(len(resamp)) #ORIGINAL

            ##If changing hanning, use hann and corresponding window var
            hann = self.create_window(start_samp, current_samp, end_samp, self.resamplesize)
            if len(hann) != len(resamp):
                window = resamp * hann[:-1]
            else:
                window = resamp * hann
                #window = resamp * hann

            ##CHECKS##
            #print "FIRST RESAMP DIGITS - ", resamp[:5]
            #print "FIRST HANNING DIGITS - ", hann[:5]
            #print "FIRST WINDOW DIGITS - ", window[:5]

            ##Ensures start and ends are 0.0, NOT -0.0
            #window[0]=0.0#abs(window[0])
            #window[self.resamplesize-2] = float(0.0)#abs(window[self.resamplesize-1])

            #print "FIRST WINDOW DIGITS - ", window[:5]
            #print "Window type -  ", type(window)
            #print "last zero type     ", type(window[self.resamplesize-1])
            #print "next to last zero type     ", type(window[self.resamplesize-2])
            #print window[0:20] #CHECK

            ##Appends the binary pitchmark flag##
            new_win = window[1:-1] #First and last are always 0, which cannot be normalised. They are added back in/accounted for at the overlap and add step.
            new_win = numpy.append(new_win, self.tracker[i][1]) ##Appends the pitchmark flag.
            if args.logstretch == True:
                new_win = numpy.append(new_win, 20*numpy.log(stretch_ratio)) ##Appends the log of the stretch ration * 20.
            else:
                new_win = numpy.append(new_win, stretch_ratio) ##Appends the bare stretch ratio

            #print "VOICING FLAG IS      ", new_win[-2]
            #print "LENGTH      ", len(new_win)

            #new_win = numpy.dtype('Float32')
            #print "APPEND FLAG - new_win type - ", type(new_win)

            """CHECKS"""
            #print start_samp,
            #print "=",

            #print end_samp
            #print "Resampled size", len(resamp)
            #print "Window size", len(window)
            #print "Window size", len(hann)
            #print "SHOULD BE", self.resamplesize
            #print "========================"

            """ADD CURRENT FRAME TO MASTER ARRAY"""
            master.append(new_win)

        # print len(master)
        #print "Number of frames ====    ", b

        """SAVES TO AN OUTFILE for reference/checks"""
        #numpy.savetxt("./numpy_out_adapt.txt", master)
        #numpy.savetxt(str(self.bare)+"_STRETCH.out", master)

        """Save to binary output file"""
        #binio.array_to_binary_file(master, str(self.bare)+".mgc")
        print "Saving ......................"
        #self.array_to_binary_file(master, str(self.bare)+"_"+str(self.resamplesize)+"_PM_rat.mgc")
        self.array_to_binary_file(master, str(self.bare)+"_filter_test"+str(args.filter[0])+".mgc")
        print "================================================================"




    def extract_segment(self, filename, start, end):
        subprocess.call(["sox", str(filename[:-3]+".wav"), "./temp.wav", "trim", str(start), str(end)])

    def extract_duration(self, filename):
        subprocess.call(["sox", --i, -D, str(filename)])

    def read(self, path):
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
            #self.wave = data[1]


    def create_window(self, start, peak, end, resampled):
        #where resampled is the resampling size

        pos = round(((peak - start)/(end-start)) * resampled)
        neg_pos = round(((end - peak) / (end-start)) * resampled)
        first=numpy.hanning(2*pos)
        second=numpy.hanning(2*neg_pos)

        new_window = numpy.append((first[:pos]), (second[neg_pos:]))
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
    #read_from_file(args.input)
    #get_time_range(tracker)
    S = Proc()
    S.whole_function(args.input)
