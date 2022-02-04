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
from scipy import signal, interpolate, optimize, stats
from scipy.io import wavfile #for wav file reading


###################################################################################
#                               AXCTD WAV FILE READING                            #
###################################################################################


def readAXCTDwavfile(inputfile, timerange):
    
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
        
    
    tstart = timerange[0]
    istart = int(fs * tstart) #AXCTD profile starts at 13:43 (length=21:43) therefore 63% of the way in
    
    #how much of audio file to select
    if timerange[1] == -1:
        iend = len(audiostream)
    else:
        iend = int(fs*timerange[1]) 
    
    if istart >= iend:
        logging.info(f"[!] Start index is after end index!\nStart={istart}, End={iend}")
        exit()
    
    # Normalize amplitude/DC offset of audio signal
    pcm_dc = np.mean(audiostream[istart:iend])
    pcm_ampl = np.max(np.abs(audiostream[istart:iend]))
    pcm = (audiostream[istart:iend].astype(np.float) - pcm_dc) / pcm_ampl
    
    
    
    # downsampling if necessary 
    if fs > 50000: 
        pcm = signal.decimate(pcm, 2)
        fs /= 2
        
        
    return pcm, tstart, fs


    
    
    
    
###################################################################################
#                               PROFILE START IDENTIFICATION                      #
###################################################################################


def identify_prof_start(pcm, tstart, fs, settings):
    
    pcm_len = len(pcm)
    
    #getting power time series for 400 Hz signals at a 2 Hz resolution using a 16-point window (initial check)
    t400a, s400a = get_prof_sig_timeseries(pcm, fs, 400, 2, 16, True)
    thresh = settings[2]
    s400ind_coarse = np.where(s400a >=thresh)[0][0] #rough estimate of 400 Hz pulse start time
    
    #precisely identifying 400 Hz pulse start using a 128 Hz resolution, 32 point window over 5 second period
    newwin = int(np.round(fs*2.5))
    s400ind_coarse_pcm = int(np.round(s400ind_coarse*fs/4)) #converting index power array to index in pcm array
    if s400ind_coarse_pcm - newwin <= 0:
        offset = 0
        pcm2 = pcm[:2*newwin]
    elif s400ind_coarse_pcm + newwin >= pcm_len: #4500 Hz pulse shouldn't be within 2.5 seconds of end of file
        logging.info("[!] Error detecting AXCTD header data")
        exit()
    else:
        offset = t400a[s400ind_coarse] - 2.5
        pcm2 = pcm[s400ind_coarse_pcm-newwin : s400ind_coarse_pcm+newwin]
    t400b,s400b = get_prof_sig_timeseries(pcm2, fs, 400, 25, 32, True)
    s400ind = np.where(s400b >= thresh)[0][0]
    t400 = t400b[s400ind] + offset
    
    #if determining prof start by time from 1st 400 Hz pulse, add specified time to ID'ed 400 Hz pulse start
    if settings[0] == "timefrompulse": 
        t7500 = t400 + settings[1]
        
    else: #otherwise- autodetect 7500 Hz tone initiation within specified window after 400 Hz pulse
        stime = t400 + settings[1][0] #range over which to search for pulse in seconds since t400
        etime = t400 + settings[1][1] 
        
        maxtime = int(np.floor(pcm_len*fs))
        if etime > maxtime:
            etime = maxtime
        elif stime >= maxtime:
            logging.info(f"[!] 400 Hz pulse detected at {t400} sec,\nspecified profile tone check starts at {stime} sec,\nbut file ends at {maxtime} sec\nAdjust start time for tone detection and rerun!")
        
        s400ind_pcm = int(np.round(s400ind*fs/4)) #converting index power array to index in pcm array
        t7500check,s7500check = get_prof_sig_timeseries(pcm[int(np.round(stime*fs)):int(np.round(etime*fs))], fs, 7500, 25, 32, False) #final checker
        s7500indlist = np.where(s7500check >= thresh)[0]
        
        if len(s7500indlist) > 0:
            t7500 = t7500check[s7500indlist[0]] + stime
        else:
            t7500 = t400 + 33
            logging.info(f"[!] Unable to identify 7500 Hz tone start within specified window, defaulting to 33 seconds after first 400 Hz pulse")
            
    #accounting for PCM data in file outside range analyzed in this function
    t400 += tstart
    t7500 += tstart
    
    logging.info(f"Identified first 400 Hz pulse at {t400} sec, profile start at {t7500} sec")
    
    return t400, t7500
    
    
    
    
        
#get time series of normalized power level at a spec. frequency    
def get_prof_sig_timeseries(pcm, fs, freq, sigfs, N, norm):
    
    trigterm = 2*np.pi*np.arange(0,N)/fs*freq #trig term for power calculation
    stimes = np.arange(0, (int(np.floor(len(pcm)-N)))/fs, 1/sigfs) #times at which each sample is collected
    
    signals = [] #stores power levels 
    for t in stimes:
        cind = int(np.round(t*fs))
        cdata = pcm[cind:cind+N]
        signals.append(np.abs(np.sum(cdata*np.cos(trigterm) + 1j*cdata*np.sin(trigterm))))
    
    #convert to array
    signals = np.asarray(signals)
    
    if norm:
        maxsig = np.max(np.abs(signals))
        signals /= maxsig #normalizing
    
    return stimes,signals

    

def trim_file_to_prof(pcm, tstart, fs, t400):
    
    if t400 > tstart:
        newstartind = int(np.round(fs*(t400-tstart)))
        tstart = t400
        pcm = pcm[newstartind:]
    
    return pcm, tstart
    
    
    
    
    


###################################################################################
#                         AXCTD PCM DATA FSK DEMODULATION                         #
###################################################################################
    

def demodulate_axctd(pcm, tstart, fs):
    """ 
    Note for the future: We could probably adapt the code here for a realtime algorithm:
    https://github.com/EliasOenal/multimon-ng/blob/master/demod_afsk12.c
    """
    
    logging.info(f"Demodulating audio stream (size {np.shape(pcm)})")
    
    #basic configuration
    f1 = 400 # bit 1 (mark) = 400 Hz
    f2 = 800 # bit 0 (space) = 800 Hz
    bitrate = 800 #symbol rate = 800 Hz
    phase_error = 25 #accountable phase error expressed as +/- X % of a single bit's period
    bit_inset = 2 #number of points after zero crossing where bit identification starts
    high_bit_scale = 1.45 #scale factor for high frequency bit to correct for erroneous high power on low frequencies
    fprof = 7500 #7500 Hz signal associated with active profile data transmission
    
    # Basic signal accounting information
    dt = 1/fs
    len_pcm = len(pcm)
    sig_time = len_pcm * dt # signal time duration
    sig_endtime = tstart + sig_time
    logging.info("PCM signal length: {:d} samples, {:0.3f} seconds"
    "".format(len_pcm, sig_time))
        
    # Use bandpass filter to extract digital data only
    sos = signal.butter(6, 1200, btype='lowpass', fs=fs, output='sos')
    pcmlow = signal.sosfilt(sos, pcm)
    
    #finding all zero crossings
    pcmsign = np.sign(pcmlow)
    pcmsign[pcmsign == 0] = 1
    zerocrossings = np.where(pcmsign[:-1] != pcmsign[1:])[0]
    
    #identify zero crossings to separate bits within phase error percentage 
    N = int(np.round(fs/bitrate*(1 - phase_error/100))) #first crossing following previous that could be a full bit
    bit_edges = [zerocrossings[0]]
    prev = zerocrossings[0]
    for c in zerocrossings:
        if c >= prev + N:
            bit_edges.append(c)
            prev = c
    
    #calculate power at FSK frequencies for each "bit"
    Nn = N - bit_inset
    trig1 = 2*np.pi*np.arange(0,Nn)/fs*f1 #trig term for power calculation
    trig2 = 2*np.pi*np.arange(0,Nn)/fs*f2
    s1 = [] #stores power levels 
    s2 = []
    
    for e in bit_edges[:-1]:
        cdata = pcmlow[e+bit_inset:e+N]
        s1.append(np.abs(np.sum(cdata*np.cos(trig1) + 1j*cdata*np.sin(trig1))))
        s2.append(np.abs(np.sum(cdata*np.cos(trig2) + 1j*cdata*np.sin(trig2)))*high_bit_scale)
        
    #determine each bit and associated confidence (power of identified freq / power of alternate freq)
    bits = []
    conf = []
    for (p1,p2) in zip(s1,s2):
        if p1 >= p2:
            bits.append(1)
            conf.append(p1/p2)
        else:
            bits.append(0)
            conf.append(p2/p1)
    
    
    #generate corresponding array of times
    tpcm = np.arange(tstart, sig_endtime, dt)
    times = tpcm[bit_edges]
    
    logging.info(f"Demodulation complete- found {len(bits)} bits, average confidence ratio={np.mean(conf):3.2f}")        
    
    return times, bits, conf

    
    
    
    
    