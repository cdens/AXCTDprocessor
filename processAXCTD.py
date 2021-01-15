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

import numpy as np
import demodulateAXCTD, parseAXCTDframes




###################################################################################
#                               AXCTD PROCESSING DRIVER                           #
###################################################################################

def processAXCTD(inputfile, outputdir, timerange=[0,-1], settings = ["autodetect",[30,36]]):
    
    #reading in audio file
    pcm, tstart, fs = demodulateAXCTD.readAXCTDwavfile(inputfile, timerange)
    
    #identifying profile tone start from file
    if settings[0] == "manual":
        t7500 = settings[1]
    else:
        t400, t7500 = demodulateAXCTD.identify_prof_start(pcm, tstart, fs, settings)
        pcm, tstart = demodulateAXCTD.trim_file_to_prof(pcm, tstart, fs, t400) #trimming PCM data to transmission only
        
    #demodulating PCM data
    times, bitstream, signallevel = demodulateAXCTD.demodulate_axctd(pcm, tstart, fs)
    
    #writing test output data to file
    with open(outputdir + '_demod.txt', 'wt') as f:
        for t, b, sig in zip(times, bitstream, signallevel):
            f.write(f"{b},{t:0.6f},{sig:0.3f}\n")
    
    #parsing bitstream to CTD profile
    logging.info("[+] Parsing AXCTD bitstream into frames")
    T, C, S, z, proftime, profframes, frames = parseAXCTDframes.parse_bitstream_to_profile(bitstream, times, t7500)
    
    #writing CTD data to ASCII file
    outputfile = outputdir + '_profile.txt'
    logging.info("[+] Writing AXCTD data to " + outputfile)
    with open(outputfile, "wt") as f:
        f.write("Depth (m)   Temperature (C)   Conductivity (mS/cm)   Salinity (PSU)\n")
        for (ct, cc, cs, cz) in zip(T, C, S, z):
            f.write(f"{cz:9.2f}{ct: 15.2f}{cc: 20.2f}{cs: 17.2f}\n")
        
    return 0


    
    
    
    
    
###################################################################################
#                           ARGUMENT PARSING + MAIN                               #
###################################################################################

#function to handle input arguments

def main():
    
    parser = argparse.ArgumentParser(description='Demodulate an audio file to text')
    parser.add_argument('-i', '--input', default='ERROR_NO_FILE_SPECIFIED', help='Input WAV file')
    parser.add_argument('-o', '--output', default='output/prof', help='Output file prefix')
    
    parser.add_argument('-s', '--starttime', default='0', help='AXCTD start time in WAV file') #13:43
    parser.add_argument('-e', '--endtime',  default='-1', help='AXCTD end time in WAV file') #20:00
    
    parser.add_argument('-m', '--mode', default='autodetect', help='Profile transmission detection mode (autodetect, timefrompulse, or manual)')
    parser.add_argument('--autodetect-start',  default='30', help='Point at which autodetect algorithm starts scanning for profile transmission start')
    parser.add_argument('--autodetect-end',  default='36', help='Point at which autodetect algorithm stops scanning for profile transmission start')
    parser.add_argument('--signal-threshold',  default='0.5', help='Threshold for normalized signal levels in profile transmission autodetection')
    
    parser.add_argument('--header-duration',  default='33', help='Duration between first 400 Hz pulse and profile start')
    parser.add_argument('-p', '--profile-start', default='33', help='Profile transmission detection mode (autodetect, timefrompulse, or manual)')
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    #configuring logging level
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout)
    np.set_printoptions(precision=4)
    
    #checking for input WAV file
    if args.input == 'ERROR_NO_FILE_SPECIFIED':
        logging.info("[!] Error- no input WAV file specified! Terminating")
        exit()
    elif not os.path.exists(args.input):
        logging.info("[!] Specified input file does not exist! Terminating")
        exit()
    
    #generating output directory
    outpath = args.output
    outdir = os.path.dirname(outpath)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    #WAV time bounds for processing
    timerange = [parse_times(args.starttime), parse_times(args.endtime)]
    if timerange[0] < 0:
        timerange[0] == 0
    if timerange[1] < 0:
        timerange[1] = -1
    
    #settings for profile transmission detection
    settings = [args.mode]
    if args.mode == "autodetect":
        settings.append([float(args.autodetect_start), float(args.autodetect_end)])
    elif args.mode == "timefrompulse":
        settings.append(float(header_duration))
    elif args.mode == "manual":
        settings.append(parse_times(args.profile_start))
    else:
        logging.info(f"[!] Specified profile transmission detector mode {settings[0]} is not a valid option. Please select 'autodetect', 'timefrompulse', 'manual'")
    
    #profile threshold
    thresh = float(args.signal_threshold)
    if thresh <= 0 or thresh > 1:
        logging.info("[!] Specified signal threshold outside the range (0,1], defaulting to 0.5")
        thresh = 0.5
    settings.append(thresh)

    return processAXCTD(args.input, outpath, timerange, settings)
    

    
    
def parse_times(time_string):
    try:
        if ":" in time_string: #format is HH:MM:SS 
            t = 0
            for i,val in enumerate(reversed(time_string.split(":"))):
                if i <= 2: #only works up to hours place
                    t += int(val)*60**i
                else:
                    logging.info("[!] Warning- ignoring all end time information past the hours place (HH:MM:SS)")
        else:
            t = int(time_string)
        return t
        
    except ValueError:
        logging.info("[!] Unable to interpret specified start time- defaulting to 00:00")
        return -2

        

#MAIN
if __name__ == "__main__":
    main()



