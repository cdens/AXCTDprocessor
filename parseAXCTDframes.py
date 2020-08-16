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

import numpy as np

# Can we remove gsw as a dependency? -> would need a different way to handle T/C -> S conversion
try:
    import gsw
    USE_GSW = True
except ImportError:
    USE_GSW = False


###################################################################################
#                               BITSTREAM PARSING                                 #
###################################################################################


def parse_bitstream_to_profile(bitstream, times, p7500, p7500thresh):
    
    proftime = []
    T = []
    C = []
    S = []
    z = []
    frames = [] # tuple of index, frame
    profframes = [] #good frames matching profile points
    triggertime = None
    
    masks = generateMasks()    
    
    #initializing fields for loop
    s = 0 #starting bit
    numbits = len(bitstream)
    trash = []
    
    #looping through bits until finished parsing
    while True:
        
        foundMatch = False
        
        if s >= numbits - 32: #won't overrun bitstream
            break
            
        #pulling current segment
        frame = bitstream[s:s+32]
        
        #verifying that frame meets requirements, otherwise increasing start bit and rechecking
        if frame[0:3] != [1, 0, 0] or not checkECC(frame, masks):
            trash.append(frame[0])
            s += 1
            continue
        
        #once a good frame has been identified, print all trash frames
        if trash:
            print_frame("               Trash", trash, s, times[s], p7500[s], None)
            frames.append((0, s, trash))
            trash = []
            
        # Check pilot tone (after ECC checks so profile triggering requires (1) sufficient P7500 and (2) a valid frame)
        if triggertime is None and p7500[s] >= p7500thresh:
            triggertime = times[s]
            logging.info(f"Triggered @ {triggertime:0.6f} sec")
        
        if triggertime is None: #profile hasn't been triggered
            frames.append((1, s, frame))
            print_frame(" Frame (Pre-trigger)", frame, s, times[s], p7500[s], None)
            
        else: #profile has been triggered
            
            #current profile time
            ctime = times[s] - triggertime
            
            #converting frame to T/C/S/z
            Tint, Cint = convertFrameToInt(frame)
            cT, cC, cS, cz = convertIntsToFloats(Tint, Cint, ctime)
            
            #storing frame/time
            frames.append((2, s, frame))
            profframes.append(frame)
            proftime.append(ctime)
            
            #storing values for profile
            T.append(cT)
            C.append(cC)
            S.append(cS)
            z.append(cz)
            
            #printing frame with profile info
            print_frame("Frame (Post-trigger)", frame, s, times[s], p7500[s], [cz,cT,cC,cS])
            
        #increase start bit by 32 to search for next frame
        s += 32

    # End parse bitstream
    return T, C, S, z, proftime, profframes, frames



def print_frame(label, frame, s, t, p7500, data):
    framestring = "".join( ['1' if b else '0' for b in frame])
    if data is None:
        msg = f"{label} s={s:7d}, t={t:12.6f} p={p7500:5.2f} {framestring:s}"
    else:
        msg = f"{label} s={s:7d}, t={t:12.6f} p={p7500:5.2f} {framestring:s} z={data[0]:07.2f}, T={data[1]:05.2f}, C={data[2]:05.2f}, S={data[3]:05.2f}"
        
    logging.debug(msg)



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

    if sum(frame)%2 != 0: #if parity isn't even
        return False
        
    data = np.asarray(frame[3:26]) #data ECC applies to
    ecc = frame[26:31] #ECC bits
    
    for (bitMasks, bit) in zip(masks, ecc): #checking masks for each ECC bit
        for mask in bitMasks: #loops through all 8 masks for each bit
            if (sum(data*mask) + bit)%2 != 0:
                return False # isValid = False
    
    return True #isValid



###################################################################################
#          FRAME CONVERSION TO TEMPERATURE/CONDUCTIVITY/SALINITY/DEPTH            #
###################################################################################

def convertFrameToInt(frame):

    Tint = binListToInt(frame[14:26])
    Cint = binListToInt(frame[3:14])
    
    return Tint, Cint
    
    
def convertIntsToFloats(Tint, Cint, time):
    
    #depth from time
    z = 0.72 + 2.76124*time - 0.000238007*time**2
    
    #temperature/conductivity from frame
    T = 0.0107164443 * Tint - 5.5387245882
    C = 0.0153199220 * Cint - 0.0622192776
    
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
        



    
