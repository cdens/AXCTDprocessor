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

def parse_bitstream_to_profile(bitstream, times, p7500, debugdir='', ecc=True):
    """ Parse a bitstream to a profile of temperature and conductivity

    """
    triggertime, frames = decode_bitstream_to_frames(bitstream, times, p7500, debugdir, ecc=ecc)
    ret = parse_frames_to_profile(frames, times, triggertime)
    # TODO: return frames
    return ret


def parse_frames_to_profile(frames, times, triggertime):
    """
    [GN] Proposal:
    change this routine to return only 3 parallel lists:
    list_time - time relative to trigger in seconds
    list_temp - decoded temperature as deg C
    list_cond - decoded conductivity as mS/cm

    Salinity can be calculated externally from return values
    and depth also calculated externally from time
    """

    # Initialize arrays
    T = [] # temperature
    C = [] # conductivity
    S = [] # salinity
    z = [] # depth
    time1 = [] # time

    # Parse frames into science data
    for type, s, frame in frames:
        if type == 0:
            continue
        t = times[s]

        # For now, don't do frames before the trigger time
        if t < triggertime:
            continue

        time1.append(t - triggertime)
        cT, cC, cS, cz = convertFrame(frame, time1[-1])
        T.append(cT)
        C.append(cC)
        S.append(cS)
        z.append(cz)
    return T, C, S, z, time1

def get_pulse_and_trigger(bitstream, minlength, times, p7500, p7500thresh):
    #identifying end of final long pulse
    # could use regexp
    triggertime = None
    last_pulse_ind = 0
    pulse_length = 0
    for i, b in enumerate(bitstream):
        if b == '1':
            pulse_length += 1
        else: # assert b == '0'
            pulse_length = 0

        if pulse_length >= minlength:
            last_pulse_ind = i

        # Check pilot tone
        if triggertime is None and p7500[i] >= p7500thresh:
            triggertime = times[i]
            logging.info(f"Triggered @ {triggertime:0.6f} sec")

    return triggertime, last_pulse_ind

def decode_bitstream_to_frames(bitstream, times, p7500, debugdir, ecc=True):
    """ Decode a bitstream to a series of frames.

    If debugdir is provided, a text debugging files are saved to this directory

    returns (triggertime, frames)
    triggertime is the detected time of the trigger
    frames a list of of tuples, where the first element is the type
    frames[i][0]: 1 = a frame, 0 = bits not used in decoding (trash)
    frames[i][1]: The start time of this frame
    frames[i][2]: The bits of this frame or of unused bits, as a string of bits

    """

    frames = [] # tuple of type, index, frame
    
    p7500thresh = 0.7 #threshold for valid data from power at 7500 Hz
    
    triggertime, last_pulse_ind = get_pulse_and_trigger(bitstream, 100, times, p7500, p7500thresh)


    #starts on final 1 bit (starting first '100' frame header)
    s = last_pulse_ind
    logging.debug("Pulse starts at s={:d} t={:0.6f}".format(s, times[s]))

    eccv = ErrorCorrection()
    if ecc:
        logging.info("Verifying ECC checksums")
    else:
        logging.info("Not verifying ECC checksums")

    #initializing fields for loop
    numbits = len(bitstream)
    # Index of the next unconsumed bit, for "trash" tracking
    nextbit_ind = s

    fdebug = None
    if debugdir:
        outfile = os.path.join(debugdir, 'bitframes.txt')
        fdebug = open(outfile, 'wt')
        logging.info("Writing " + outfile)

    #looping through bits until finished parsing
    while s + 32 < numbits:
        # TODO: we should really error correct these bits too
        if bitstream[s:s+3] != '100':
            s += 1
            continue

        frame_b = bitstring_to_int(bitstream[s:s+32])
        frame_parity = eccv.parity_b(frame_b)
        frame_ecc = eccv.check_b(frame_b)

        if ecc and not frame_parity:  # if frame ecc is incorrect and requested, skip to next bit
            s += 1
            continue

        if nextbit_ind < s:
            msg = format_frame('Trash', bitstream[nextbit_ind:s], nextbit_ind, times[nextbit_ind], False, False, triggertime)
            #logging.debug(msg)
            if fdebug:
                fdebug.write(msg + "\n")
            frames.append((0, nextbit_ind, bitstream[nextbit_ind:s]))
            #nextbit_ind = s

        msg = format_frame('Frame', bitstream[s:s+32], s, times[s], frame_parity, frame_ecc, triggertime)
        #logging.debug(msg)
        if fdebug:
            fdebug.write(msg + "\n")

        frames.append((1, s, bitstream[s:s+32]))
        s += 32
        nextbit_ind = s

    if nextbit_ind < s:
        msg = format_frame('Trash', bitstream[nextbit_ind:s], nextbit_ind, times[nextbit_ind], False, False, triggertime)
        #logging.debug(msg)
        if fdebug:
            fdebug.write(msg + "\n")
        frames.append((0, nextbit_ind, bitstream[nextbit_ind:s]))
        nextbit_ind = s


    # End parse bitstream
    if fdebug:
        fdebug.close()


    return triggertime, frames



def format_frame(label, cseg, s, t, frame_parity, frame_ecc, triggertime):
    """ Output type, frame time, index, length, parity ok, ecc ok, frame contents """
    #sseg = "".join( ['1' if b else '0' for b in cseg])
    t2 = t - triggertime
    if label == 'Frame':
        token_parity = 'PAR_OK ' if frame_parity else 'PAR_BAD'
        token_ecc    = 'ECC_OK ' if frame_parity else 'ECC_BAD'
        token_hex    = "{:08x}".format(int(cseg, 2))
    else:
        token_parity = '-      '
        token_ecc    = '-      '
        token_hex = '--------'
    return "{:s} t={:12.6f}, t'={:12.3f}, s={:7d}, len={:3},{:s},{:s},{:s},{:s}" \
           "".format(label, t, t2, s, len(cseg), token_parity, token_ecc, token_hex, cseg)



###################################################################################
#                   ERROR CORRECTING CODE CHECK/APPLICATION                       #
###################################################################################

class ErrorCorrection:

    def __init__(self):
        self.masks = generateMasks()
        self.bitmasks = get_masks()

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

    def parity_b(self, frame_b):
        """ Check only parity of this frame. Returns true if parity is correct (even),
        or false if incorrect (odd)  """
        return bitparity(frame_b) == 0 #if parity isn't even

    def check_b(self, frame_b):
        """  RUN ECC CHECK ON FRAME WITH MASKS CREATED IN GENERATEMASKS()
        Note that currently, the bits are internally stored backwards, where bit 32
        (the last bit received in the bitstream frame) is in the lowest order bit position
        frame_b is an integer representing the frame
        """

        if bitparity(frame_b) != 0: #if parity isn't even
            return False

        data_b = (frame_b >> 5) & 0x007fffff # frame[4:27], data bits
        ecc_b = (frame_b >> 0) & 0x1f # frame[27:32], ECC bits

        for ii, bitmasks in enumerate(self.bitmasks): # Check masks for each ECC bit
            parity = 1 if ecc_b & (1 << (4-ii)) else 0
            for mask in bitmasks:
                if bitparity(data_b & mask) != parity:
                    return False

        return True



    def correct(self, frame):
        """ Correct a frame """
        pass

# see also: https://wiki.python.org/moin/BitManipulation

def bitcount(w):
    """ Count the number of 1 bits in a 32-bit word
    Glenn Rhoads snippets of c
    http://gcrhoads.byethost4.com/Code/BitOps/bitcount.c
    """
    w = (0x55555555 & w) + ((0xaaaaaaaa & w) >> 1)
    w = (0x33333333 & w) + ((0xcccccccc & w) >> 2)
    w = (0x0f0f0f0f & w) + ((0xf0f0f0f0 & w) >> 4)
    w = (0x00ff00ff & w) + ((0xff00ff00 & w) >> 8)
    w = (0x0000ffff & w) + ((0xffff0000 & w) >>16)
    return w

def bitparity(w):
    # http://gcrhoads.byethost4.com/Code/BitOps/parity.c
    w ^= w>>1;
    w ^= w>>2;
    w ^= w>>4;
    w ^= w>>8;
    w ^= w>>16;
    return w & 1;



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
        
        
def test_ecc():
    print("test_ecc()")
    ecc = ErrorCorrection()
    
    # Increment by a prime number for speed
    for ii, x in enumerate(range(0, 0xffffffff, 2011)):
        if ii % 1000 == 0:
            print("ecc:", x)
        y = intToBinList(x, 32)
        assert ecc.check(y) == ecc.check_b(x)



###################################################################################
#          FRAME CONVERSION TO TEMPERATURE/CONDUCTIVITY/SALINITY/DEPTH            #
###################################################################################

def convertFrame(frame, time):
    
    #depth from time
    z = 0.72 + 2.76124*time - 0.000238007*time**2
    
    #temperature/conductivity from frame as an integer
    t_int = bitstring_to_int(frame[15:27])
    c_int = bitstring_to_int(frame[ 4:15])

    T = 0.0107164443 * t_int - 5.5387245882
    C = 0.0153199220 * c_int - 0.0622192776
    
    
    #salinity from temperature/conductivity/depth
    if USE_GSW:
        S = gsw.SP_from_C(C,T,z) #assumes pressure (mbar) approx. equals depth (m)
    else:
        S = float('nan')
    logging.debug(f"{time:0.3f} {t_int} {T} ; {c_int} {C}")
    
    return T, C, S, z





###################################################################################
#                         BINARY LIST / INTEGER CONVERSION                       #
###################################################################################

def bitstring_to_int(bitstring):
    """ Convert a bitstring to an integer with the leftmost bit
    as the most significant bit """
    #return int(bitstring[::-1], 2)
    return int(bitstring, 2)

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


def test_intbin():
    """ Check that these two functions are inverses of each other """
    for masklen in range(1, 32):
        maxv = 2 << masklen
        if masklen < 20:
            inc = 7
        elif masklen < 28:
            inc = 1009
        else:
            inc = 10099
        print("Mask length:", masklen)
        for x in range(0, 0xffffffff, inc):
            if x >= maxv:
                break
            listbits = intToBinList(x, 32)
            y = binListToInt(listbits)
            assert x == y

        


def main():
    test_intbin()
    test_ecc()


if __name__ == "__main__":
    main()
