#! /usr/bin/python2.7
#PASS PITCHMARK FILE and corresponding wav file TO THIS FUNCTION AND DURATION(from the parent shell script)
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
parser.add_argument('--samplerate', '-s', help="Samplerate")
parser.add_argument('--duration', '-d', help="Duration of wav file")

args = parser.parse_args()
#####################################################

class Proc():
    def __init__(self):
        self.tracker=[]
        self.samplerate = float(args.samplerate) #48000 #SET SAMPLERATE
        #self.wav_ref = args.wav_reference

    def whole_function(self, input_file):
        data = self.read_from_file(input_file)
        self.get_time_ranges(data)



    def read_from_file(self, filename):
        #print str(filename)
        #print "worked"
        self.filename = filename

        #tracker = []
        for line in open(filename):
            current = line.split()
            try:
                current = [float(x) for x in current]
                self.tracker.append(current)
            except:
                pass

    #print tracker
        return self.tracker


    def get_time_ranges(self, tracker):

        #self.read(self.wav_ref)
        master=[]
        #print "[",

        maximum=0

        for (i, x) in enumerate(self.tracker):
            #print type(i)
            #print i
            #print x
            self.current = i
            if self.tracker[i][1] == 0: #IF NO PITCHMARK
                if i == 0:
                    self.beginning = 0.00
                #elif i+1 == len(self.tracker):
                #    break
                else:
                    self.beginning = self.tracker[i-1][0]
            else: #PITCHMARK PRESENT
                if i == 0:
                    self.beginning = 0.00
                #elif i+1 == len(self.tracker):
                #    break
                else:
                    self.beginning = self.tracker[i-1][0]
            if i == len(self.tracker)-1:
                self.end = float(args.duration)
            else:
                self.end = self.tracker[i+1][0]
                    #return (beginning, end)
            print self.beginning,
            print "---",
            print self.end
            print "="
            # THIS IS THE BULK OF THE WORK


            #round(time*samplerate)
            start_samp = round(self.beginning*self.samplerate) #PLUS/MINUS 1?? INDEXING AN ARRAY - but already index 0 at time 0...
            end_samp = round(self.end*self.samplerate)

            length = end_samp - start_samp
            if length > maximum:
                maximum = length

            """CHECKS"""
            print start_samp,
            print "=",
            print end_samp

            print "LENGTH IS", length

            print "========================"
            #frame = self.wave[start_samp:end_samp]
            #print len(frame)
            #print len(self.wave)

            #resamp = scipy.signal.resample(frame, len(frame)) #Second argument can change based on max size.
            #window = resamp * numpy.hanning(len(resamp))
            #print window[0:20],



            #master.append(window[0:20])
        #print master
        print "Maximum length is", maximum

        #numpy.savetxt("./numpy_out.txt", master)

        #print "]"
        #print master


    """
            #THIS IS THE CULNKY WAV EXTRACTION METHOD#
            self.extract_segment(self.filename, self.beginning, self.end) #creates temp.wav
            self.read("./temp.wav")
            resamp = scipy.signal.resample(self.wave, len(self.wave)) #Second argument can change based on max size.
            wind = resamp * numpy.hanning(len(resamp))
            print wind[0:20] #+ "\n" #SHORTENED FOR TEST
            #SET TO RETURN FOR ACTUAL RUN
            #print "================="
    """


#UP TO HERE Works

    def extract_segment(self, filename, start, end):
        subprocess.call(["sox", str(filename[:-3]+".wav"), "./temp.wav", "trim", str(start), str(end)])

    def extract_duration(self, filename):
        subprocess.call(["sox", --i, -D, str(filename)])

    def read(self, path):
        data = wf.read(path)
        self.wave = data[1]

if __name__ == "__main__":
    #read_from_file(args.input)
    #get_time_range(tracker)
    S = Proc()
    S.whole_function(args.input)
