
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
#parser.add_argument('--duration', '-d', help="Duration of wav file")
parser.add_argument('--preemphasis', '-p', action='store_true', help="Add a preemphasis filter. Set to coeff=0.95")
parser.add_argument('--filter', '-f', nargs='*', help='Select a filter - lowpass, highpass, bandpass and values')

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
        #self.time = self.extract_duration(args.wav_reference)
        #self.resamplesize = 250 #SET RESAMPLING SIZE

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
                print "#####################################"
                data = self.read_from_file(self.pm_fol+pm_fn)
                self.get_time_ranges(data)
                bottom = np.min(np.where(self.all != 0)) - 1

        print "lowest bound is = ", bottom
        top = np.max(np.where(self.all != 0)) + 1
        print "highest bound is = ", top
        new_dimensionality = top - bottom
        print "new dimensionality is = ", new_dimensionality
        print "#########"

        print "frame count is = ", self.fr_count
        print "mean frame length = ", (self.fr_total / float(self.fr_count))

        master_peak = self.centre - bottom
        print "master peak is = ", master_peak

        print "max frame info"
        print "sentence = ", self.max_frame[0], "  frame = ", self.max_frame[1], "  length = ", self.max_frame[2]
        print "start = ", self.max_frame[3], "  current = ", self.max_frame[4], "  end = ", self.max_frame[5]
        print "voicing = ", self.max_frame[6]

        #plt.plot(self.all)
        # plt.show()


                ##ORIGINAL
                #data = self.read_from_file(input_file)
                #self.get_time_ranges(data)

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

        b = 0
        self.read(self.wav_fn)
        master=[]
        timestamp_list=[]
        new_tracker = []
        for value in tracker:
            new_tracker.append(value[0])

        #if args.logstretch == True:
        #    print "Logstretch selected"

        for j in np.arange(0.000, float(self.time), 0.005):

            #Find nearest timestamp for each timestep
            time = min(self.timestamp_list, key=lambda x:abs(x-j))
            #print "time is = ", time
            ar = np.where(np.array(new_tracker) == time)
            #print "ar is = ", ar
            i = int(ar[0])
            #try:
                #i = int(ar[0])
            #except:
                #print "##EXCEPTION## CHECK"
                #for k in ar[0]:
                    #print k, " corresponds to time ", self.timestamp_list[k]
                #i = int(ar[0][0])
            #print "so i is = ", i

            """CHECKS"""
            #print "j is === ", j
            #print x
            #print "NEAREST TIMESTAMP IS === ", time
            #print "AS AN INDEX IS ===  ", ar
            #print "i is ====   ", i
            #print "##########################"

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
                self.end = float(self.time)
            else:
                self.end = self.tracker[i+1][0]


            start_samp = round(self.beginning*self.samplerate) #PLUS/MINUS 1?? INDEXING AN ARRAY - but already index 0 at time 0...
            current_samp = round(self.current*self.samplerate)
            end_samp = round(self.end*self.samplerate)
            peak = current_samp - start_samp
            #PRINT CHECSK


            frame = self.wave[start_samp:end_samp]



            print "###############"
            print "Start samp = ", start_samp, "current samp = ", current_samp, "peak   = ", peak, "end sampe = ", end_samp, "length", len(frame)

            #peak = len(self.wave)

            """APPLYING WINDOW"""
            ##If static hanning window
            #window = resamp * numpy.hanning(len(resamp)) #ORIGINAL

            ##If changing hanning, use hann and corresponding window var
            hann = self.create_window(start_samp, current_samp, end_samp, len(frame))
            if len(hann) != len(frame):
                window = frame * hann[:-1]
            else:
                window = frame * hann
                #window = resamp * hann

            if len(frame) > self.max_frame[2]:
                self.max_frame = [str(self.pm_fn), i, len(frame), self.beginning, self.current, self.end, self.tracker[i][1]]
                #plt.close()
                #plt.plot(frame)
                #plt.plot(window)
            else:
                pass

            ##INITIALISE ADD FRAME
            add_position = self.centre - peak
            new_frame = np.append(np.zeros(add_position), window)
            add_frame = np.append(new_frame, (np.zeros(len(self.all) - len(new_frame))))
            #complete = np.append(add_frame, self.tracker[i][1])

            #self.store[str(self.pm_fn[:-2])] = {i : [add_frame, self.tracker[i][1]]}

            self.fr_count += 1
            self.fr_total += len(window)


            self.all = (self.all + add_frame)/1.5
            #self.all[self.all != 0 ] = 1

        #print np.where(self.all == 0)[0]

        ##TRIM##


        """
        # print len(master)
        #print "Number of frames ====    ", b

        #SAVES TO AN OUTFILE for reference/checks"""
        #numpy.savetxt("./numpy_out_adapt.txt", master)
        #numpy.savetxt(str(self.bare)+"_STRETCH.out", master)

        """Save to binary output file"""
        #binio.array_to_binary_file(master, str(self.bare)+".mgc")
        #print "Saving ......................"
        #self.array_to_binary_file(master, str(self.bare)+"_"+str(self.resamplesize)+"_PM_rat.mgc")
        #self.array_to_binary_file(master, str(self.bare)+"_filter_test"+str(args.filter[0])+".mgc")
        #print "================================================================"




    def extract_segment(self, filename, start, end):
        subprocess.call(["sox", str(filename[:-3]+".wav"), "./temp.wav", "trim", str(start), str(end)])

    def extract_duration(self, filename):
        process = subprocess.Popen(["sox", "--i", "-D", str(filename)], stdout=subprocess.PIPE)
        time, err = process.communicate()
        return time
        #time = subprocess.call(["sox", "--i", "-D", str(filename)])
        #return time

    def read(self, path):
        data = wf.read(path)

        if args.preemphasis == True:
            print "Preemphasis selected"
            self.wave = sigproc.preemphasis(data[1])
        else:
            print "No preemphasis"
            self.wave = data[1]

        if args.filter == True:

            if args.filter[0] == 'bandpass':
                print "Bandpass selected = ", args.filter[1], " - ", args.filter[2]
                top = float(args.filter[2])/self.samplerate
                bottom = float(args.filter[1])/self.samplerate
                b, a = scipy.signal.ellip(2, 5, 40, [bottom, top], btype='bandpass')
                self.wave = scipy.signal.filtfilt(b, a, data[1])
                #pass

            elif args.filter[0] == 'lowpass':
                print "Lowpass selected - ", args.filter[1]
                b, a = scipy.signal.ellip(2, 5, 40, float(args.filter[1])/self.samplerate, btype='lowpass')
                self.wave = scipy.signal.filtfilt(b, a, data[1])

            elif args.filter[0] == 'highpass':
                print "Highpass selected", args.filter[1]
                b, a = scipy.signal.ellip(2, 5, 40, float(args.filter[1])/self.samplerate, btype='highpass')
                self.wave = scipy.signal.filtfilt(b, a, data[1])

            else:
                print "argument not recognised"

        else:
            self.wave=data[1]





    def create_window(self, start, peak, end, resampled):
        #where resampled is the resampling size

        pos = round(((peak - start)/(end-start)) * resampled)
        neg_pos = round(((end - peak) / (end-start)) * resampled)
        first=np.hanning(2*pos)
        second=np.hanning(2*neg_pos)

        new_window = np.append((first[:pos]), (second[neg_pos:]))
        return new_window


#######################################################################################
########### Taken from binary_io.py from the DNN toolkit ##############################
    def array_to_binary_file(self, data, output_file_name):
        data = np.array(data, 'float32')

        fid = open(output_file_name, 'wb')
        data.tofile(fid)
        fid.close()

#######################################################################################

if __name__ == "__main__":
    #read_from_file(args.input)
    #get_time_range(tracker)
    S = Proc()
    S.whole_function(args.input)
