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
import gsw


###################################################################################
#                               BITSTREAM PARSING                                 #
###################################################################################

def parseBitstreamToProfile(bitstream, times, p7500):
    
    T = []
    C = []
    S = []
    z = []
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
        else:
            
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
                    
                        cT, cC, cS, cz = convertFrame(cseg[m:m+32], times[s+m]-triggertime)
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
                
    return T, C, S, z



    
    
    

###################################################################################
#                   ERROR CORRECTING CODE CHECK/APPLICATION                       #
###################################################################################



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
        
        
    
    
#  RUN ECC CHECK ON FRAME WITH MASKS CREATED IN GENERATEMASKS()
def checkECC(frame, masks):

    isValid = True
    
    if sum(frame)%2 != 0: #if parity isn't even
        return False
        
    data = np.asarray(frame[4:27]) #data ECC applies to
    ecc = frame[27:32] #ECC bits
    
    for (bitMasks, bit) in zip(masks, ecc): #checking masks for each ECC bit
        for mask in bitMasks: #loops through all 8 masks for each bit
            if (sum(data*mask) + bit)%2 != 0:
                isValid = False
    
    return isValid


    
    

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
    S = gsw.SP_from_C(C,T,z) #assumes pressure (mbar) approx. equals depth (m)
    
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
        



    





