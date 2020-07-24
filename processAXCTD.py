#! /usr/bin/env python3

#   Purpose: process data from AXCTD audio files
#   Usage: via command line: -i flag specifies input audio (WAV) file and -o
#           specifies output text file containing AXCTD profile
#       e.g. "python3 processAXCTD -i inputaudiofile.wav -o outputASCIIfile.txt"
#
#   This file is a part of AXCTDprocessor
#
#    AXCTDprocessor in this file is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    AXCTDprocessor is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with AXCTDprocessor.  If not, see <https://www.gnu.org/licenses/>.
#


###################################################################################
#                                    IMPORTS                                      #
###################################################################################

import argparse
import logging
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for wav file reading

import demodulateAXCTD, parseAXCTDframes


###################################################################################
#                               AXCTD PROCESSING DRIVER                           #
###################################################################################

def processAXCTD(inputfile, outputdir, plot=False, fromAudio=True):

    outputfile = os.path.join(outputdir, 'output.txt')
    demodfile = os.path.join(outputdir, 'demodbitstream.txt')

    if fromAudio:
        #reading WAV file
        print("[+] Reading audio file")
        fs, snd = wavfile.read(inputfile)
        
        #if multiple channels, sum them together
        sndshape = np.shape(snd) #array size (tuple)
        ndims = len(sndshape) #number of dimensions
        if ndims == 1: #if one channel, use that
            audiostream = snd
        elif ndims == 2: #if two channels
            #audiostream = np.sum(snd,axis=1) #sum them up
            audiostream = snd[:][0] #use channel 1
            print(np.shape(snd))
            print(len(audiostream))
            exit()
        else:
            print("[!] Error- unexpected number of dimensions in audio file- terminating!")
            exit()
        
        logging.info("Demodulating audio")
        #demodulating PCM data
        times, bitstream, signallevel, p7500, figures = demodulateAXCTD.demodulate_axctd(audiostream, fs, plot)

        if plot:
            plt.show()

        #writing test output data to file
        with open(demodfile, 'wt') as f:
            for t, b, sig, prof in zip(times, bitstream, signallevel, p7500):
                f.write(f"{b},{t:0.6f},{prof:0.3f},{sig:0.3f}\n")
    else:
        bitstream = []
        times = []
        p7500 = []
        with open(demodbitstream, 'rt') as f:
            for line in f:
                line = line.strip().split(',')
                bitstream.append(int(line[0]))
                times.append(float(line[1]))
                p7500.append(float(line[2]))
        times = np.asarray(times)
        p7500 = np.asarray(p7500)
    
    #parsing bitstream to CTD profile
    print("[+] Parsing AXCTD bitstream into frames")
    T,C,S,z = parseAXCTDframes.parseBitstreamToProfile(bitstream, times, p7500)
    
    #writing CTD data to ASCII file
    print("[+] Writing AXCTD data to " + outputfile)
    with open(outputfile, "wt") as f:
        f.write("Depth (m)\tTemperature (C)\tConductivity (mS/cm)\tSalinity (PSU)\n")
        for (ct, cc, cs, cz) in zip(T, C, S, z):
            f.write(f"{cz:6.1f}\t\t{ct:6.2f}\t\t{cc:6.2f}\t\t{cs:6.2f}\n")





###################################################################################
#                           ARGUMENT PARSING + MAIN                               #
###################################################################################

#function to handle input arguments

def main():
    parser = argparse.ArgumentParser(description='Demodulate an audio file to text')
    parser.add_argument('-i', '--input', default='testfiles/sample_full.wav')
    parser.add_argument('-o', '--output', default='testfiles', help='output directory')
    parser.add_argument('--plot', action="store_true", help='Show plots')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    np.set_printoptions(precision=4)

    return processAXCTD(args.input, args.output, plot=args.plot)


#MAIN
if __name__ == "__main__":
    main()



