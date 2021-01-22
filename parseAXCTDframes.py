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

import os
import sys
import logging
import argparse
from collections import namedtuple

import numpy as np

try:
    import gsw
    USE_GSW = True
except ImportError:
    USE_GSW = False


###################################################################################
#                               BITSTREAM PARSING                                 #
###################################################################################


def parse_bitstream_to_profile(bitstream, times, t7500):
    
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
        if frame[0:2] != [1, 0] or not checkECC(frame, masks):
            trash.append(frame[0])
            s += 1
            continue
        
        #once a good frame has been identified, print all trash frames
        if trash:
            print_frame("               Trash", trash, s, times[s], None)
            frames.append((0, s, trash))
            trash = []
            
        # Check pilot tone (after ECC checks so profile triggering requires (1) sufficient P7500 and (2) a valid frame)
        if times[s] <= t7500:
            frames.append((1, s, frame))
            print_frame(" Frame (Pre-trigger)", frame, s, times[s], None)
            
        else: #profile has been triggered
            
            #current profile time
            ctime = times[s] - t7500
            
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
            print_frame("Frame (Post-trigger)", frame, s, times[s], [cz,cT,cC,cS])
            
        #increase start bit by 32 to search for next frame
        s += 32

    # End parse bitstream
    return T, C, S, z, proftime, profframes, frames



def print_frame(label, frame, s, t, data):
    framestring = "".join( ['1' if b else '0' for b in frame])
    if data is None:
        msg = f"{label} s={s:7d}, t={t:12.6f} {framestring:s}"
    else:
        msg = f"{label} s={s:7d}, t={t:12.6f} {framestring:s} z={data[0]:07.2f}, T={data[1]:05.2f}, C={data[2]:05.2f}, S={data[3]:05.2f}"
        
    logging.debug(msg)


#  GENERATING ECC MASKS FOR FRAME PARSING  

def get_masks():
    #8 masks per ECC bit
    mask_ints = [[3166839,3167863,3168887,3169911,7360887,7361911,7362935,7363959], #bit 27
            [553292,554316,555340,556364,4747852,4748876,4749900,4750924], #bit 28
            [274854,275878,276902,277926,4469414,4470438,4471462,4472486], #bit 29
            [2233171,2234195,2235219,2236243,6426707,6427731,6428755,6429779], #bit 30
            [86494,87518,88542,89566,4281054,4282078,4283102,4284126]] #bit 31
    return mask_ints

def generateMasks():
    maskLen = 23 #each mask is 23 bits long
    
    mask_ints = get_masks()
    masks = []
    
    for cMaskInts in mask_ints:
        cmasks = []
        for cInt in cMaskInts: 
            cmasks.append(np.asarray(intToBinList(cInt, maskLen)))
        masks.append(cmasks)
    
    return masks
        
        
    

#  RUN ECC CHECK ON FRAME WITH MASKS CREATED IN GENERATEMASKS()
def checkECC(frame, masks):

    #if sum(frame)%2 != 0: #if parity isn't even
    #    return False
        
    #data = binListToInt(frame[:27])
    #ecc = binListToInt(frame[27:])
    #
    #if ecc != 0:
    #    if data/ecc == 101:
    #        return True
    #    else:
    #        return False
    
    
        
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
    """ Convert a frame to integer fields
    frame is a list of bits """
    Tint = binListToInt(frame[14:26])
    Cint = binListToInt(frame[2:14])
    
    return Tint, Cint
    
    
def convertIntsToFloats(Tint, Cint, time):
    """ Convert a list of integer data fields to observations (floats) """
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
    """ Convert a list of bits into a binary number """
    x = 0
    mask = 0x1 << (len(binary) - 1)
    for b in binary:
        if b:
            x |= mask
        mask >>= 1

    return x


def intToBinList(cInt, masklen):
    """ Convert a number into a list of bits with length
    at least masklen  """
    x = cInt
    i = 0
    bin_list = [0] * masklen
    while x:
        try:
            bin_list[i] = x & 1
        except IndexError: # if there are more bits than masklen
            bin_list.append(x & 1)
        x >>= 1
        i += 1

    bin_list.reverse()
    return bin_list

    
            
