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

import logging
import os

import numpy as np

# Can we remove gsw as a dependency?
try:
    import gsw
    USE_GSW = True
except ImportError:
    USE_GSW = False


###################################################################################
#                               BITSTREAM PARSING                                 #
###################################################################################

def parseBitstreamToProfile(bitstream, times, p7500):
    
    timeout = []
    T = []
    C = []
    S = []
    z = []
    frames = [] # tuple of index, frame
    triggered = False
    
    p7500thresh = 0.7 #threshold for valid data from power at 7500 Hz
    masks = generateMasks()
    


    #identifying end of final pulse
    lastPulseInd = 0
    sumConnectedOnes = 0
    for i,b in enumerate(bitstream):
        if b == 1:
            sumConnectedOnes += 1
        else:
            sumConnectedOnes = 0
            
        if sumConnectedOnes >= 100:
            lastPulseInd = i
                
    s = lastPulseInd #starts on final 1 bit (starting first '100' frame header)
    
    
    #initializing fields for loop
    numbits = len(bitstream)
    isDone = False
    
    #looping through bits until finished parsing
    while not isDone:
        
        foundMatch = False
        
        if s >= numbits - 45: #won't overrun bitstream
            isDone = True
            break

        s -= 2 #back up 2 bits in case one was skipped
        cseg = bitstream[s:s+40] #grabbing current segment (40 bits -> -2 to 38) to analyze
            
        #looking for frame start matches
        matchinds = []
        for i in range(16):
            if cseg[i:i+3] == [1,0,0]:
                matchinds.append(i)
                   
        #trim matches to only get value where ECC conditions are met
        for m in matchinds:
            if checkECC(cseg[m:m+32], masks):
                foundMatch = True
                
                if p7500[s+m] >= p7500thresh: #if profile signal detected, ID/append points
                    
                    if not triggered:
                        triggered = True
                        triggertime = times[s+m]
                        print(f"Triggered @ {times[s+m]}s")
                    timeout.append(times[s+m]-triggertime)
                    cT, cC, cS, cz = convertFrame(cseg[m:m+32], timeout[-1])
                    T.append(cT)
                    C.append(cC)
                    S.append(cS)
                    z.append(cz)
                    
                break #stop checking subsequent matches
            
        #jump to next frame if valid one was identified, otherwise skip forward 16 bits and check again
        if foundMatch:
            s += m + 32
        else:
            s += 16

    return T, C, S, z, timeout

def parse_bitstream_to_profile(bitstream, times, p7500, debugdir=''):
    """ Parse a bitstream to a profile of temperature and conductivity

    """

    timeout = []
    T = []
    C = []
    S = []
    z = []

    triggertime, frames = decode_bitstream_to_frames(bitstream, times, p7500, debugdir)

    # Parse frames into science data
    for type, s, frame, t in frames:
        if type == 0:
            continue

        # For now, don't do frames before the trigger time
        if t < triggertime:
            continue

        timeout.append(t - triggertime)
        cT, cC, cS, cz = convertFrame(frame, timeout[-1])
        T.append(cT)
        C.append(cC)
        S.append(cS)
        z.append(cz)
    # TODO: return frames

    return T, C, S, z, timeout



def decode_bitstream_to_frames(bitstream, times, p7500, debugdir):
    """ Decode a bitstream to a series of frames.

    If debugdir is provided, a text debugging files are saved to this directory

    returns (triggertime, frames)
    triggertime is the detected time of the trigger
    frames a list of of tuples, where the first element is the type
    frames[i][0]: 1 = a frame, 0 = bits not used in decoding
    frames[i][1]: The start time of this frame
    frames[i][2]: The bits of this frame or unused bits

    """


    
    frames = [] # tuple of index, frame
    triggertime = None
    
    p7500thresh = 0.7 #threshold for valid data from power at 7500 Hz
    ecc = ErrorCorrection()
    
    # Convert bitstream to a string of characters
    #bitstream = ''.join(['1' if b else '0' for b in bitstream0])

    #identifying end of final long pulse
    last_pulse_ind = 0
    pulse_length = 0
    for i, b in enumerate(bitstream):
        if b == 1:
            pulse_length += 1
        else:
            pulse_length = 0
        if pulse_length >= 100:
            last_pulse_ind = i


        # Check pilot tone
        if triggertime is None and p7500[i] >= p7500thresh:
            triggertime = times[i]
            logging.info(f"Triggered @ {triggertime:0.6f} sec")


    #starts on final 1 bit (starting first '100' frame header)
    s = last_pulse_ind
    logging.debug("Pulse starts at s={:d} t={:0.6f}".format(s, times[s]))

    #initializing fields for loop
    numbits = len(bitstream)
    trash = []
    
    #looping through bits until finished parsing
    while s + 32 < numbits:
        if bitstream[s:s+3] != [1, 0, 0]:
            trash.append(bitstream[s])
            s += 1
            continue

        if not ecc.check(bitstream[s:s+32]):
            trash.append(bitstream[s])
            s += 1
            continue

        if trash:
            logging.debug(format_frame("Trash", trash, s, times[s]))
            frames.append((0, s, trash, times[s - len(trash)]))
            trash = []

        frames.append((1, s, bitstream[s:s+32], times[s]))
        logging.debug(format_frame("Frame", bitstream[s:s+32], s, times[s]))
        s += 32

    # End parse bitstream

    if debugdir:
        outfile = os.path.join(debugdir, 'bitframes.txt')
        logging.info("Writing " + outfile)
        with open(outfile, 'wt') as fout:
            for type, s, frame, t in frames:
                label = "Frame" if type == 1 else "Trash"
                msg = format_frame(label, frame, s, t) + "\n"
                fout.write(msg)


    return triggertime, frames



def format_frame(label, cseg, s, t):
    sseg = "".join( ['1' if b else '0' for b in cseg])
    msg = "{:s} t={:12.6f}, s={:7d}, len={:3}, {:s}".format(label, t, s,len(sseg), sseg)
    return msg



###################################################################################
#                   ERROR CORRECTING CODE CHECK/APPLICATION                       #
###################################################################################

class ErrorCorrection:

    def __init__(self):
        self.masks = generateMasks()

    def check(self, frame):
        """  RUN ECC CHECK ON FRAME WITH MASKS CREATED IN GENERATEMASKS() """

        if sum(frame)%2 != 0: #if parity isn't even
            return False

        data = np.asarray(frame[4:27]) #data ECC applies to
        ecc = frame[27:32] #ECC bits

        for (bitMasks, bit) in zip(self.masks, ecc): #checking masks for each ECC bit
            for mask in bitMasks: #loops through all 8 masks for each bit
                if (sum(data*mask) + bit)%2 != 0:
                    return False # isValid = False
        return True #isValid


    def correct(self, frame):
        """ Correct a frame """
        pass

#  GENERATING ECC MASKS FOR FRAME PARSING  
def generateMasks():
    
    maskLen = 23 #each mask is 23 bits long
    
    #8 masks per ECC bit
    maskInts = [[3166839,3167863,3168887,3169911,7360887,7361911,7362935,7363959], #bit 27
            [553292,554316,555340,556364,4747852,4748876,4749900,4750924], #bit 28
            [274854,275878,276902,277926,4469414,4470438,4471462,4472486], #bit 29
            [2233171,2234195,2235219,2236243,6426707,6427731,6428755,6429779], #bit 30
            [86494,87518,88542,89566,4281054,4282078,4283102,4284126]] #bit 31
    
    masks = []
    
    for cMaskInts in maskInts:
        cmasks = []
        for cInt in cMaskInts: 
            cmasks.append(np.asarray(intToBinList(cInt, maskLen)))
        masks.append(cmasks)
    
    return masks
        
        
    
    



###################################################################################
#          FRAME CONVERSION TO TEMPERATURE/CONDUCTIVITY/SALINITY/DEPTH            #
###################################################################################

def convertFrame(frame, time):
    
    #depth from time
    z = 0.72 + 2.76124*time - 0.000238007*time**2
    
    #temperature/conductivity from frame
    T = 0.0107164443 * binListToInt(frame[15:27]) - 5.5387245882
    C = 0.0153199220 * binListToInt(frame[4:15]) - 0.0622192776
    
    #salinity from temperature/conductivity/depth
    if USE_GSW:
        S = gsw.SP_from_C(C,T,z) #assumes pressure (mbar) approx. equals depth (m)
    else:
        S = float('nan')
    
    return T, C, S, z





###################################################################################
#                         BINARY LIST / INTEGER CONVERSION                       #
###################################################################################

def binListToInt(binary):
    
    intVal = 0
    
    for i,b in enumerate(reversed(binary)):
        intVal += b*2**i

    return intVal

    
    
    
def intToBinList(cInt, maskLen):

    #increasing mask length if it's too small for the number
    while cInt >= 2**maskLen:
        maskLen += 1
    
    
    binList = []
    for i in range(maskLen):
        cExp = maskLen-i-1
        if cInt - 2**cExp >= 0:
            cInt -= 2**cExp
            binList.append(1)
        else:
            binList.append(0)
    
    return binList
        



    
