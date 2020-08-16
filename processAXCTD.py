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

def processAXCTD(inputfile, outputdir, timerange=[0,-1], plot=False):
    
    demodfile = outputdir + '_demod.txt'
    bitfile = outputdir + '_bitstream.txt'
    outputfile = outputdir + '_profile.txt'

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
          
    #writing test output data to file
    with open(demodfile, 'wt') as f:
        for t, b, sig, prof in zip(times, bitstream, signallevel, p7500):
            f.write(f"{b},{t:0.6f},{prof:0.3f},{sig:0.3f}\n")
            
    #writing bitstream to file (50 columns long)
    with open(bitfile, 'w') as f:
        lb = len(bitstream)
        i = 0
        il = 0
        bpline = 50
        while i < lb:
            f.write(f"{bitstream[i]}")
            i +=  1
            il += 1
            if il >= bpline:
                f.write("\n")
                il = 0
    
    #saving all figures
    for ii, fig in enumerate(figures):
        filename = os.path.join(outputdir, f"{ii+1}_demod.png")
        fig.savefig(filename)
        logging.info("Saving " + filename)

    
    #parsing bitstream to CTD profile
    logging.info("[+] Parsing AXCTD bitstream into frames")
    T, C, S, z, proftime, profframes, frames = parseAXCTDframes.parse_bitstream_to_profile(bitstream, times, p7500)
    
    #writing CTD data to ASCII file
    logging.info("[+] Writing AXCTD data to " + outputfile)
    with open(outputfile, "wt") as f:
        f.write("Depth (m)   Temperature (C)   Conductivity (mS/cm)   Salinity (PSU)\n")
        for (ct, cc, cs, cz) in zip(T, C, S, z):
            f.write(f"{cz:9.2f}{ct: 15.2f}{cc: 20.2f}{cs: 17.2f}\n")

    if plot:
        fig1, axs1 = plt.subplots(1, 3, sharey=True)
        axs1[0].plot(T, z, label='Temp (C)')
        axs1[1].plot(C, z, label='Cond (mS/cm)')
        axs1[2].plot(S, z, label='Sal (PSU)')
        axs1[0].set_ylabel('Depth (m)')
        axs1[0].set_xlabel('Temperature (C)')
        axs1[1].set_xlabel('Conductivity (mS/cm)')
        axs1[2].set_xlabel('Salinity (PSU)')
        for ax in axs1:
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.grid(True)
        plt.tight_layout()

        logging.info("Saving " + filename)
        fig1.savefig(os.path.join(outputdir, "ctd.png"))

    if plot:
        logging.info("Displaying plots (close to terminate program)")
        plt.show()
        
    return 0


###################################################################################
#                           ARGUMENT PARSING + MAIN                               #
###################################################################################

#function to handle input arguments

def main():
    parser = argparse.ArgumentParser(description='Demodulate an audio file to text')
    parser.add_argument('-i', '--input', default='testfiles/sample_full.wav', help='Input WAV file')
    parser.add_argument('-o', '--output', default='testfiles/test', help='Output file header')
    parser.add_argument('-s', '--starttime', default='0', help='AXCTD start time in WAV file') #13:43
    parser.add_argument('-e', '--endtime',  default='-1', help='AXCTD end time in WAV file') #20:00
    parser.add_argument('--plot', action="store_true", help='Show plots')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    #configuring logging level
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout)

    np.set_printoptions(precision=4)
    
    #reading range of WAV file to parse
    timerange = [0,-1] #default
    
    #start time
    try:
        if ":" in args.starttime: #format is HH:MM:SS 
            s = 0
            for i,val in enumerate(reversed(args.starttime.split(":"))):
                if i <= 2: #only works up to hours place
                    s += int(val)*60**i
                else:
                    logging.info("[!] Warning- ignoring all start time information past the hours place (HH:MM:SS)")
                    
            timerange[0] = s
            
        else:
            timerange[0] = int(args.starttime)
            
    except ValueError:
        logging.info("[!] Unable to interpret specified start time- defaulting to 00:00")
        
    #end time
    try:
        if ":" in args.endtime: #format is HH:MM:SS 
            e = 0
            for i,val in enumerate(reversed(args.endtime.split(":"))):
                if i <= 2: #only works up to hours place
                    e += int(val)*60**i
                else:
                    logging.info("[!] Warning- ignoring all end time information past the hours place (HH:MM:SS)")
                    
            timerange[1] = e
        else:
            timerange[1] = int(args.endtime)
        
    except ValueError:
        logging.info("[!] Unable to interpret specified end time- defaulting to end of file")

    return processAXCTD(args.input, args.output, timerange, plot=args.plot)


#MAIN
if __name__ == "__main__":
    main()



