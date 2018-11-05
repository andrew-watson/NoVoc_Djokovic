
import sys
import numpy as np
import scipy
import argparse
import itertools
import os
import subprocess
import scipy.io.wavfile as wf
import scipy.signal
from features import sigproc
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='Framing, resampling, windowing.')
parser.add_argument('--input', '-i', help="Pitchmark folder")
parser.add_argument('--wav_reference', '-w', help="Reference Wav folder")
parser.add_argument('--samplerate', '-s', help="Samplerate")

args = parser.parse_args()
#########################################################

class Proc():
    def __init__(self):
        self.tracker=[]
        self.samplerate = float(args.samplerate) #8000 #SET SAMPLERATE
        self.pm_fol = str('./'+str(args.input)+'/')
        self.wav_fol = str('./'+str(args.wav_reference)+'/')
        self.all = np.zeros(1000)
        self.centre = round(len(self.all) / 2.0)
        self.max_frame = [0, 0, 0]
        self.fr_count = 0
        self.fr_total = 0

        self.store = {}

    def whole_function(self, input_file):

        for pm_fn in os.listdir(self.pm_fol):
            if pm_fn.endswith(".pm"):
                self.pm_fn = pm_fn
                self.wav_fn = str(self.wav_fol)+str(pm_fn[:-2])+'wav'
                print pm_fn
                print self.wav_fn
                if os.path.isfile(pm_fn):
                    print pm_fn
                    print self.wav_fn
                self.time = self.extract_duration(self.wav_fn)
                #print self.time
                print "###########"
                data = self.read_from_file(self.pm_fol+pm_fn)
                self.get_time_ranges(data)
                bottom = np.min(np.where(self.all != 0)) - 1

        print "################################################"
        print "lowest bound is = ", bottom
        top = np.max(np.where(self.all != 0)) + 1
        print "highest bound is = ", top
        new_dimensionality = top - bottom
        print "new dimensionality is = ", new_dimensionality
        print "################################################"

        print "frame count is = ", self.fr_count
        print "mean frame length = ", (self.fr_total / float(self.fr_count))

        master_peak = self.centre - bottom
        print "master peak is = ", master_peak

        print "################################################"
        print "Max Frame Info"
        print "sentence = ", self.max_frame[0], "  frame = ", self.max_frame[1], "  length = ", self.max_frame[2]
        print "start = ", self.max_frame[3], "  current = ", self.max_frame[4], "  end = ", self.max_frame[5]
        print "voicing = ", self.max_frame[6]


    def read_from_file(self, filename):
        self.filename = filename
        self.bare = filename[:-3]
        self.timestamp_list = []
        self.tracker = []

        with open(filename) as f:
            for line in f:
                current = line.split()
                try:
                    current = [float(x) for x in current]
                    self.tracker.append(current)
                    self.timestamp_list.append(current[0])
                except:
                    pass
            #print self.timestamp_list
        return self.tracker


    def get_time_ranges(self, tracker):
        i=0
        j=0

        self.read(self.wav_fn)
        master=[]
        timestamp_list=[]
        new_tracker = []
        for value in tracker:
            new_tracker.append(value[0])

        for j in np.arange(0.000, float(self.time), 0.005):

            #Find nearest timestamp for each timestep
            time = min(self.timestamp_list, key=lambda x:abs(x-j))
            #print "time is = ", time
            ar = np.where(np.array(new_tracker) == time)
            #print "ar is = ", ar
            i = int(ar[0])

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
                self.end = float(self.time)
            else:
                self.end = self.tracker[i+1][0]

            start_samp = round(self.beginning*self.samplerate)
            current_samp = round(self.current*self.samplerate)
            end_samp = round(self.end*self.samplerate)
            peak = current_samp - start_samp

            frame = self.wave[int(start_samp):int(end_samp)]

            if len(frame) > self.max_frame[2]:
                self.max_frame = [str(self.pm_fn), i, len(frame), self.beginning, self.current, self.end, self.tracker[i][1]]
            else:
                pass

            ##INITIALISE ADD FRAME
            add_position = int(self.centre - peak)
            new_frame = np.append(np.zeros(add_position), frame)
            add_frame = np.append(new_frame, (np.zeros(len(self.all) - len(new_frame))))

            self.fr_count += 1
            self.fr_total += len(frame)

            self.all = (self.all + add_frame)/1.5
            #self.all[self.all != 0 ] = 1

    def extract_duration(self, filename):
        process = subprocess.Popen(["sox", "--i", "-D", str(filename)], stdout=subprocess.PIPE)
        time, err = process.communicate()
        return time


    def read(self, path):
        data = wf.read(path)
        self.wave=data[1]



#######################################################################################

if __name__ == "__main__":
    #read_from_file(args.input)
    #get_time_range(tracker)
    S = Proc()
    S.whole_function(args.input)
