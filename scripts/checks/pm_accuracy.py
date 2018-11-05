
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
import random

parser = argparse.ArgumentParser(
    description='Framing, resampling, windowing.')
parser.add_argument('--generated', '-g', default="GEN", help="GEN folder")
parser.add_argument('--reference', '-o', default="REF", help="ref folder")
parser.add_argument('--dimensionality', '-d', default=250, help="Dimensionality")
#parser.add_argument('--plot', '-p', action='store_true', help="Plot the error and absolute values")

args = parser.parse_args()
#########################################################


def read_features(input_file):
    pms = []
    for (i, x) in enumerate(input_file):
        pms.append(x[-2])
    return pms

def calculate_error(generated, reference):
    error_vals = []
    for (i, x) in enumerate(generated):
        err = reference[i] - generated[i]
        #print err
        #type(err)
        error_vals.append(err)
    return error_vals

def calculate_accuracy(generated, reference, verbose):
    correct = 0
    incorrect = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for (i, x) in enumerate(generated):
        #print i, x
        if generated[i] < 1 and reference[i] == 0:
            TN+=1
            classification_result.append('green')
        elif generated[i] < 1 and reference[i] == 1:
            FN+=1
            classification_result.append('red')
        elif generated[i] >=1 and reference[i] == 1:
            TP+=1
            classification_result.append('green')
        elif generated[i] >=1 and reference[i] == 0:
            FP+=1
            classification_result.append('red')
        else:
            print "Something is broken"
            break
    accuracy = float(TP + TN) / (TP + TN + FP + FN)
    if verbose == True:
        print "True Positives = ", TP
        print "False Negatives = ", FN
        print "False Positives = ", FP
        print "True Negatives = ", TN

        print "Accuracy = ", accuracy

    return TP, FP, FN, TN

def open_binary_file(file_name, dimensionality):
    mgcfile = open(file_name, 'rb')
    contents = np.fromfile(mgcfile, dtype=np.float32)
    mgcfile.close()
    assert contents.size % float(dimensionality) == 0.0,'specified dimension %s not compatible with data'%(dimensionality)
    contents = contents[:(dimensionality * (contents.size / dimensionality))]
    contents = contents.reshape((-1, dimensionality))
    return contents

## LET'S GO TO WORK ##


pm_record = []
gen_record = []
pm_error = []

gen_folder = str('./'+str(args.generated)+'/')
ref_folder = str('./'+str(args.reference)+'/')

classification_result = []


for gen in os.listdir(gen_folder):
    if gen.endswith('.mgc'):
        gen = gen_folder+os.path.basename(gen)
        #print gen
        ref = ref_folder+os.path.basename(gen)
        #print ref
        gen_contents = open_binary_file(gen, args.dimensionality)
        ref_contents = open_binary_file(ref, args.dimensionality)

        gen_pms = read_features(gen_contents)
        ref_pms = read_features(ref_contents)

        #Calculate Accuracy for each line
        #print "Accuracy of this line = "
        calculate_accuracy(gen_pms, ref_pms, False)

        #Add to overall totals
        pm_record.extend(ref_pms)
        gen_record.extend(gen_pms)
    else:
        pass

print "============================="
print "Calculating and plotting error "
pm_error =  calculate_error(gen_record, pm_record)
plt.hist(pm_error, 40)
plt.title("Histogram of error values")
plt.show()
#plt.savefig('error_hist.png')
#plt.close()

print "============================="
print "Calculating and plotting classification accuracy"
print "============================="

class_error = calculate_accuracy(gen_record, pm_record, True)
plt.scatter(pm_record, gen_record, color=classification_result)
plt.title("Scatterplot of predicted pitchmark values")
plt.xlabel("Pitchmark Value")
plt.ylabel("Generated Value")
plt.show()

print "============================="
print "Plotting with Jitter on PM axis for clarity"
print "============================="

pm_jitter = pm_record + np.random.uniform(-0.05, 0.05, len(pm_record))

plt.scatter(pm_jitter, gen_record, color=classification_result)
plt.title("Scatterplot of predicted pitchmark values with jitter")
plt.xlabel("Pitchmark Value")
plt.ylabel("Generated Value")
plt.show()
