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

def processAXCTD(inputfile, outputdir, timerange=[0,-1], p7500thresh=10, plot=False):
    
    outputpfile = outputdir + '_p7500vtime.txt'

    #reading WAV file
    logging.info("[+] Reading audio file")
    fs, snd = wavfile.read(inputfile)
    
    #if multiple channels, sum them together
    sndshape = np.shape(snd) #array size (tuple)
    ndims = len(sndshape) #number of dimensions
    if ndims == 1: #if one channel, use that
        logging.debug("Audio file has one channel")
        audiostream = snd
    elif ndims == 2: #if two channels
        logging.debug("Audio file has multiple channels, processing data from first channel")
        #audiostream = np.sum(snd,axis=1) #sum them up
        audiostream = snd[:,0] #use first channel
    else:
        logging.info("[!] Error- unexpected number of dimensions in audio file- terminating!")
        exit()
    
    logging.info(f"Demodulating audio stream (size {np.shape(audiostream)})")
    
    #demodulating PCM data
    times, bitstream, signallevel, p7500, figures = \
          demodulateAXCTD.demodulate_axctd(audiostream, fs, timerange, plot=plot)
    
    print("identifying profile start")
    tind = [int(t*fs) for t in times[5:-5]]
    maxt = 10/fs
    tt = np.arange(0,maxt,1/fs)
    y7500 = np.cos(2*np.pi*tt*7500) + 1j*np.sin(2*np.pi*tt*7500)
    p7500v2 = [np.sum(np.abs(y7500*audiostream[ctind-5:ctind+5])) for ctind in tind]
    print("Finished IDing profile start")
          
    #writing test output data to file
    with open(outputpfile, 'wt') as f:
        for t, b, sig, prof in zip(times[5:-5], bitstream[5:-5], signallevel[5:-5], p7500v2[5:-5]):
            f.write(f"{b},{t:0.6f},{prof:0.3f},{sig:0.3f}\n")

    
            
    return 0


###################################################################################
#                           ARGUMENT PARSING + MAIN                               #
###################################################################################

#function to handle input arguments

def main():
    
    #configuring logging level
    loglevel = logging.DEBUG
    logging.basicConfig(level=loglevel, stream=sys.stdout)
    
    np.set_printoptions(precision=4)
    
    
    #execute function
    return processAXCTD('testfiles/sample_full.wav', 'testfiles/test', [821, 1200], 10.0, plot=False)


#MAIN
if __name__ == "__main__":
    main()



