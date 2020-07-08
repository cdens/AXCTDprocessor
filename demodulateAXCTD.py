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

import numpy as np



###################################################################################
#                         AXCTD FSK DEMODULATION ROUTINE                          #
###################################################################################

def demodulateAXCTD(pcm, fs):
    
    #basic configuration
    dt = 1/fs
    f1 = 400 #bit 1 = 400 Hz
    f2 = 800 #bit 0 = 800 Hz
    bitrate = 800 #symbol rate = 800 Hz
    fprof = 7500 #7500 Hz signal associated with active profile data transmission
    pctchecksig = 0.95 #percent of each segment to check
    
    #demodulation parameter calculation
    pointsperbit = int(np.round(fs/bitrate)) #number of PCM datapoints in each bit
    window = int(np.round(pctchecksig*pointsperbit))
    tt = np.arange(window)/fs
        
    #test signals for individual point FT
    y1 = np.cos(2*np.pi*f1*tt) + 1j*np.sin(2*np.pi*f1*tt)
    y2 = np.cos(2*np.pi*f2*tt) + 1j*np.sin(2*np.pi*f2*tt)
    yprof = np.cos(2*np.pi*fprof*tt) + 1j*np.sin(2*np.pi*fprof*tt)
    
    #box-smoothed PCM signal to identify/track zero crossings
    pcmlow = boxSmooth(pcm, int(np.round(3*fs/fprof)))
    
    #initializing loop
    isDone = False
    siglen = len(pcm)
    bitnum = np.floor(siglen/pointsperbit)
    start = 0 #start checking data at first point
    p7500 = []
    bitstream = []
    times = []
    c = 0
    
    #loop until end of PCM data
    while not isDone:
        
        if start < siglen - 2*pointsperbit: #won't overrun PCM data- stops ~2 bits short 
        
            #time
            times.append(start/fs)    
        
            #calculating power at each frequency
            cpcm = pcm[start:start+window]
            S1 = np.abs(np.trapz(y1*cpcm, dx=dt))
            S2 = np.abs(np.trapz(y2*cpcm, dx=dt))
            p7500.append(np.abs(np.trapz(yprof*cpcm, dx=dt)))
            
            #appending bit to stream based on which is more powerful
            if S1 > S2:
                cbit = 1
            else:
                cbit = 0
            bitstream.append(cbit)
                
            #look for upcrossings/downcrossings in expected ranges
            bitranges = [[0.9,1.1],[0.8,1.2],[0.7,1.3],[0.4,1.6]]
            for br in bitranges:
                upcoming = pcmlow[start:start+int(np.round(br[1]*pointsperbit))]
                ind = getnextncrossings(upcoming, 2-cbit, int(np.round(br[0]*pointsperbit)))
                
                if ind:
                    break
            
            #increase start point
            if not ind:
                ind = pointsperbit
            start += ind
            
            c += 1
            if c%1E4 == 0:
                print(f"\r[+] Demodulating file, progress: {int(100*c/bitnum):2d}%",end=" ")
                
        else:
            isDone = True
            
    
    #convert 7500 Hz signals to ndarray, smooth
    p7500 = boxSmooth(np.asarray(p7500), 50)
    print("\r[+] Demodulating file, progress: 100%")
    
    return bitstream, times, p7500


    



###################################################################################
#                         ZERO CROSSING IDENTIFICATION                            #
###################################################################################

def getnextncrossings(data, ncrossings, minind):
    
    ind = 0 #if checks fail- return no zero crossings found
    
    try:
        
        
        #getting indices corresponding to all upcrossings/downcrossings
        allcrossings = np.sort(np.concatenate((np.where((data[:-1] < 0) & (data[1:] >= 0))[0], 
            np.where((data[:-1] >= 0) & (data[1:] < 0))[0] )))
        
        #removing all jumps back and forth across 0 within 6 points of one another
        allcrossings = np.delete(allcrossings, np.diff(allcrossings,append=allcrossings[-1]+10) <= 6)
        
        #1 crossing expected: identifying first zero crossing within valid range
        if ncrossings == 1:
            allcrossings = allcrossings[allcrossings >= minind] #within range
            ind = allcrossings[0]
        
        #2 crossings expected: getting second valid crossing, must be within range and first crossing must be within 0.5*range    
        elif ncrossings == 2:
            allcrossings = allcrossings[allcrossings >= minind/2] #first crossing within 0.5*range
            allcrossings = [c for (i,c) in enumerate(allcrossings) if (i >= 2 and c >= minind)] #get second crossing (**now list not ndarray)
            ind = allcrossings[0]
    
    except IndexError:
        pass
    
    return ind
            

    
    
    
###################################################################################
#                                   BOX SMOOTHER                                  #
###################################################################################
    

def boxSmooth(y, npts):
    box = np.ones(npts)/npts
    return np.convolve(y, box, mode='same')
    
    
    
    