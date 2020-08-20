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
        x1, x2 = int(cseg, 2), int(cseg[::-1], 2)
        token_hex    = "{:08X} {:08X}".format(x1, x2)
    else:
        token_parity = '-      '
        token_ecc    = '-      '
        token_hex = '-------- --------'
        # cut last frame into 32-bit words
        delim = "\n" + ' ' * len("Trash t=    0.010625, t'=      -1.289, s=      0, len= 40,-      ,-      ,-------- --------,")
        s2 = delim.join(cseg[i:i+32] for i in range(0, len(cseg), 32))
        cseg = s2
        

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

def unpack_frame_b(frame_b):
    #t_int = bitstring_to_int(frame[16:26])
    #t_int = (frame_b >> 16) & 0x3ff
    # after some reverse engineering
    t_int = (frame_b >> 6) & 0x3ff  # frame[6:16]
    #t_int = (frame_b >> (32-6)) & 0x1ff # frame 6:15]
    #logging.debug("frame={0:08x} {0:032b} {0:5d}".format(frame_b))
    #logging.debug("t_int={0:03x} {0:09b} {0:5d}".format(t_int))
    
    #c_int = bitstring_to_int(frame[ 5:14])
    c_int = (frame_b >> 18) & 0x7ff # frame[28:18] (bit notation)
    logging.debug("frame={0:08x} {0:032b} {0:5d}".format(frame_b))
    logging.debug("c_int={0:03x} {0:09b} {0:5d}".format(c_int))

    return t_int, c_int

def convert_frame(time, t_int, c_int):
    
    #depth from time
    z = 0.72 + 2.76124*time - 0.000238007*time**2
    
    T = 0.0107164443 * t_int - 5.5387245882
    C = 0.0153199220 * c_int - 0.0622192776


    #salinity from temperature/conductivity/depth
    if USE_GSW:
        S = gsw.SP_from_C(C,T,z) #assumes pressure (mbar) approx. equals depth (m)
    else:
        S = float('nan')
    logging.debug(f"{time:0.3f} {t_int} {T} ; {c_int} {C}")
    
    return T, C, S, z

    

def convertFrame(frame, time):
    
    #depth from time
    z = 0.72 + 2.76124*time - 0.000238007*time**2
    
    #temperature/conductivity from frame as an integer
    t_int = bitstring_to_int(frame[16:26])
    c_int = bitstring_to_int(frame[ 5:14])

    #T = 0.0107164443 * t_int - 5.5387245882
    #C = 0.0153199220 * c_int - 0.0622192776
    T = t_int
    C = c_int
    
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


def parse_xctd_debug(infile):

    # 348 9C6888BC 9C688928 10011100011010001000100010111100 10011100011010001000100100101000 00 00
    # 349 9C688813 9C6C88E7 10011100011010001000100000010011 10011100011011001000100011100111 00 00
    #     10011100100000001000100000001101       0      //           0          //          //     544    
    #     1824     0.28344871     0.28344871     0.28344871      27.881318      27.881318      27.894723
    fieldnames = 'n hex1 hex2 bin1 bin2 f5 f6 bin3 f7 f8 f9 f10 f11 t_int c_int t1 t2 t3 c1 c2 c3'
    nfields = len(fieldnames.split())
    XCTDRawFrame = namedtuple('XCTDRawFrame', fieldnames)
    logging.info("Reading " + infile)
    state = 'skip'
    with open(infile, 'rt') as fin:
        for jj, line in enumerate(fin):
            line = line.strip()
            if state == 'skip':
                if '**** Raw XCTD Frames' in line:
                    state = 'save'
                continue
            assert state == 'save'
            fields = line.strip().split()
            if not fields:
                continue

            parse_err = False
            fields[0] = int(fields[0]) # convert to int
            if len(fields) >= 15:
                for ii in (13, 14):
                    try:
                        fields[ii] = int(fields[ii])
                    except ValueError:
                        parse_err = True
                        pass
            if len(fields) >= 21:
                for ii in range(15, 21):
                    try:
                        fields[ii] = float(fields[ii])
                    except ValueError:
                        parse_err = True
                        pass
                        fields[ii] = float('nan')

            # Add enough fields so that things will work.
            if len(fields) < nfields:
                fields.extend([None] * (nfields - len(fields)))

            if parse_err:
                logging.error("Error parsing line {:d}: {:s}".format(jj+1, line))

            yield XCTDRawFrame._make(fields)

            
def main():

    parser = argparse.ArgumentParser(description='Test frame parsing algorithms')
    parser.add_argument('-i', '--input', help='Input xctd_debug file')
    parser.add_argument('-o', '--output', default='testfiles', help='output directory')
    parser.add_argument('--test', action="store_true", help='Perform self test')
    #parser.add_argument('--plot', action="store_true", help='Show plots')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout)

    np.set_printoptions(precision=4)

    if args.test:
        test_intbin()
        test_ecc()
        return 0;

    for jj, rec in enumerate(parse_xctd_debug(args.input)):
        recnum = rec.n
        logging.debug(f"[{recnum:5d}] Start")
        try:
            frame1 = int(rec.bin3, 2)
            int(rec.t_int)
        except (TypeError, ValueError):
            # parsing error
            print(rec)
            continue
        unpack3 = unpack_frame_b(frame1)
        unpack2 = unpack_frame_b(int(rec.hex2, 16))
        print(rec)

        nomatch = True

        # Check correspondence between hex and bin
        hex1, hex2 = int(rec.hex1, 16), int(rec.hex2, 16)
        bin1, bin2, bin3 = int(rec.bin1, 2), int(rec.bin2, 2), int(rec.bin3, 2)


        #print(f"hex1={hex1:08X} hex2={hex2:08X} bin1={bin1:08X} bin2={bin2:08X} bin3={bin3:08X}")
        assert hex1 == bin1
        assert hex2 == bin2

        """ 
        frame1 = int(rec.bin3, 2)
        for bitwidth in range(8, 16):
            for bit0 in range(0, 32 - bitwidth):
                # mask off field
                mask = (1 << (bitwidth+1)) - 1
                x = (frame1 >> bit0) & mask

                if x == rec.c_int:
                    reversestr = '   ' #reversestr = "rev" if reverse else "fwd"
                    print(f"[{recnum:5d}] Matched {frame1:032b} {frame1:08X} mask={mask:08X} frame1[{bit0:d}:{bitwidth+bit0:d}]"
                          f" {rec.c_int:09b}=0x{rec.c_int:03x}={rec.c_int:d} {reversestr:s}")
                    nomatch = False
        if isinstance(rec.c_int, int) and nomatch:
            print("No Match {:08X} frame1[{:d}:{:d}] 0x{:x}={:d} {:s}"
                  "".format(frame1, 0, 0, rec.c_int, rec.c_int, ''))
        """

        print(f'[{recnum:5d}]  - unpack3: {unpack1} {rec.t_int} {rec.c_int}')
        assert unpack3[0] == rec.t_int and unpack3[1] == rec.c_int 
        print(f'[{recnum:5d}]  - unpack2: {unpack2} {rec.t_int} {rec.c_int}')
        #assert unpack2[0] == rec.t_int and unpack2[1] == rec.c_int 


        # fields to floats
        T, C, S, z =  convert_frame(0, *unpack1)
        print(f'[{recnum:5d}] unpack1 {T:08f} {C:08f} {S:08f} {z}')
        T, C, S, z =  convert_frame(0, *unpack2)
        print(f'[{recnum:5d}] unpack2 {T:08f} {C:08f} {S:08f} {z}')


if __name__ == "__main__":
    main()
