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
import numpy.fft
from features import sigproc
#import binary_io


parser = argparse.ArgumentParser(
    description='Framing, resampling, windowing.')
parser.add_argument('--input', '-i', help="Pitchmark file")
parser.add_argument('--wav_reference', '-w', help="Reference Wav file")
parser.add_argument('--duration', '-d', help="Duration of wav file")
parser.add_argument('--samplerate', '-s', help="Sample rate")
parser.add_argument('--preemphasis', '-p', action='store_true', help="Add a preemphasis filter. Set to coeff=0.95")
parser.add_argument('--logstretch', '-l', action='store_true', help='Encodes the scaled log of the stretch factor - (multiplied by 20)')
parser.add_argument('--filter', '-f', default="None", nargs='*', help='Select a filter - lowpass, highpass, bandpass and values')
parser.add_argument('--noise', '-n', action='store_true', help="Store the energy components of the top 4 frequency bands when at 16000hz")

args = parser.parse_args()
#####################################################

class Proc():
    def __init__(self):
        self.tracker=[]
        self.samplerate = float(args.samplerate) #8000 #SET SAMPLERATE
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

            #print j, "  ==========  ", i

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

            start_samp = round(self.beginning*self.samplerate) #PLUS/MINUS 1?? INDEXING AN ARRAY - but already index 0 at time 0...
            current_samp = round(self.current*self.samplerate)
            end_samp = round(self.end*self.samplerate)

            frame = self.wave[start_samp:end_samp]

            ##Storing the stretch ratio
            frame_length = len(frame)

            new_win = []
            if args.noise == True :
                #print "Noisebands being used"
                spectrum = numpy.absolute(numpy.fft.rfft(frame, n=1024))
                index_4 = round(4000.0 / (self.samplerate / len(spectrum)))
                index_5 = round(5000.0 / (self.samplerate / len(spectrum)))
                index_6 = round(6000.0 / (self.samplerate / len(spectrum)))
                index_7 = round(7000.0 / (self.samplerate / len(spectrum)))
                index_8 = round(8000.0 / (self.samplerate / len(spectrum)))
                nb_4_5 = numpy.mean(spectrum[index_4:index_5]) / 1024.0
                nb_5_6 = numpy.mean(spectrum[index_5:index_6]) / 1024.0
                nb_6_7 = numpy.mean(spectrum[index_6:index_7]) / 1024.0
                nb_7_8 = numpy.mean(spectrum[index_7:index_8]) / 1024.0
                new_win = numpy.append(new_win, nb_4_5)
                new_win = numpy.append(new_win, nb_5_6)
                new_win = numpy.append(new_win, nb_6_7)
                new_win = numpy.append(new_win, nb_7_8)


            """ADD CURRENT FRAME TO MASTER ARRAY"""
            #print new_win
            master.append(new_win)
            #print "================================================================"

        # print len(master)
        #print "Number of frames ====    ", b
        if args.noise == True :
            print "Noisebands being used"
            print "Final dimensionality is = ", len(new_win)
        else:
            print "Final dimensionality is = ", len(new_win)

        """SAVES TO AN OUTFILE for reference/checks"""
        #numpy.savetxt("./numpy_out_adapt.txt", master)
        #numpy.savetxt(str(self.bare)+"_STRETCH.out", master)

        """Save to binary output file"""
        #binio.array_to_binary_file(master, str(self.bare)+".mgc")
        print "Saving ......................"
        #self.array_to_binary_file(master, str(self.bare)+".mgc")
        #self.array_to_binary_file(master, str(self.bare)+"_"+str(self.resamplesize)+"_PM_rat.mgc")
        #self.array_to_binary_file(master, str(self.bare)+"_filter_test"+str(args.filter[0])+".mgc")
        print "================================================================"



    def read(self, path):
        data = wf.read(path)

        if args.filter == True:
            print "Filter true"

            if args.filter[0] == 'bandpass':
                print "Bandpass selected = ", args.filter[1], " - ", args.filter[2]
                top = float(args.filter[2])/self.samplerate
                bottom = float(args.filter[1])/self.samplerate
                b, a = scipy.signal.ellip(2, 5, 40, [bottom, top], btype='bandpass')
                self.wave = scipy.signal.filtfilt(b, a, data[1])
                pass

            elif args.filter[0] == 'lowpass':
                print "Lowpass selected - ", args.filter[1]
                b, a = scipy.signal.ellip(2, 5, 40, float(args.filter[1])/self.samplerate, btype='lowpass')
                self.wave = scipy.signal.filtfilt(b, a, data[1])

            elif args.filter[0] == 'highpass':
                print "Highpass selected", args.filter[1]
                b, a = scipy.signal.ellip(2, 5, 40, float(args.filter[1])/self.samplerate, btype='highpass')
                self.wave = scipy.signal.filtfilt(b, a, data[1])

            #elif args.filter[0] == 'None':
            #    "#NoFilter"
            #    self.wave = data[1]
            else:
                print "PROBLEM WITH FILTER ARGUMENT"
        else:
            print "Filter false"
            print "#NoFilter"
            self.wave = data[1]

        if args.preemphasis == True:
            print "Preemphasis selected"
            self.wave = sigproc.preemphasis(data[1])
        else:
            print "No preemphasis"
            #self.wave = data[1]


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
