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
    phase_error = 30 #accountable phase error expressed as +/- X % of a single bit's period
    bit_inset = 2 #number of points after zero crossing where bit identification starts
    high_bit_scale = 1.4 #scale factor for high frequency bit to correct for erroneous high power on low frequencies
    fprof = 7500 #7500 Hz signal associated with active profile data transmission
    
    debug = True
    
    if debug:
        pcm = pcm[int(1*fs):int(11*fs)] #cut out 10 seconds of prof transmission
        #pcm = pcm[:int(10*fs)] #cut to first 10 seconds
    
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
    
    if debug:
        np.savetxt("../DemodDebug/pcm.txt",pcmlow)
        np.savetxt("../DemodDebug/zerocross.txt",zerocrossings)
        np.savetxt("../DemodDebug/bitedges.txt",np.asarray(bit_edges))
        np.savetxt("../DemodDebug/bits.txt",np.asarray(bits))
        np.savetxt("../DemodDebug/conf.txt",np.asarray(conf))
        exit()
    
    return times, bits, conf

    
    
    
    
    
    
    
    

def demodulate_axctd_old(pcm, tstart, fs):
    """ 
    Note for the future: We could probably adapt the code here for a realtime algorithm:
    https://github.com/EliasOenal/multimon-ng/blob/master/demod_afsk12.c
    """
    
    logging.info(f"Demodulating audio stream (size {np.shape(pcm)})")
    
        
    #basic configuration
    f1 = 400 # bit 1 (mark) = 400 Hz
    f2 = 800 # bit 0 (space) = 800 Hz
    bitrate = 800 #symbol rate = 800 Hz
    fprof = 7500 #7500 Hz signal associated with active profile data transmission
    squelch_snr = 20.0 # db
    
    
    # Basic signal accounting information
    dt = 1/fs
    len_pcm = len(pcm)
    sig_time = len_pcm * dt # signal time duration
    sig_endtime = tstart + sig_time
    logging.info("PCM signal length: {:d} samples, {:0.3f} seconds"
    "".format(len_pcm, sig_time))
    
    #generate corresponding array of times
    t1 = np.arange(tstart, sig_endtime, dt)
    
    #correcting length issues if they arise (occasionally off by 1 point)
    if len(t1) > len(pcm):
        t1 = t1[:len(pcm)]
    elif len(pcm) > len(t1): 
        pcm = pcm[:len(t1)]
    
    
    # TODO: make this filter configurable
    # Use bandpass filter to extract digital data only
    #sos = signal.butter(6, [f1 * 0.5, f2 * 1.5], btype='bandpass', fs=fs, output='sos')
    #sos = signal.butter(6, [5, 1200], btype='bandpass', fs=fs, output='sos')
    sos = signal.butter(6, 1200, btype='lowpass', fs=fs, output='sos')
    #sos = signal.cheby1(6, 0.1, 1200, btype='lowpass', fs=fs, output='sos')
    pcmlow = signal.sosfilt(sos, pcm)

    # figures = []
    #--------------------------------------------------------------
    logging.debug("Making spectrogram")


    t2, data_db, active_db = signal_levels(pcm, fs=fs, fprof=fprof)
    t2 += tstart


    # Squelch digital data and display
    f_datasq = interpolate.interp1d(t2, data_db, bounds_error=False, fill_value=0)
    f_active = interpolate.interp1d(t2, active_db, bounds_error=False, fill_value=0)

    squelch_mask_t1 = (f_datasq(t1) > squelch_snr)
    pcm_sq = pcmlow * squelch_mask_t1


    logging.debug("Done filtering")
    # Perform complex IQ downconversion, demodulation

    fc = (f1 + f2) / 2
    # convert original lowpassed signal to complex domain IQ, centered at fc

    # downconvert and lowpass filter
    fcarrier = np.exp(2*np.pi*1j*-fc*t1)
    # TODO: make the downconversion filter frequency configurable
    # if doquad:
    #     sosdcx = signal.butter(9, fc*0.5, btype='lowpass', fs=fs, output='sos')
    # else:
    sosdcx = signal.butter(4, fc*0.6, btype='lowpass', fs=fs, output='sos')

    pcmiq = signal.sosfilt(sosdcx, fcarrier * pcmlow)
    pcmiq /= np.max(np.abs(pcmiq))
    pcmiq *= squelch_mask_t1

    

    fftopts = {'fs': fs, 'nfft': 16384, 'scaling': 'spectrum', 'return_onesided': False}

    f, Pxx = signal.periodogram(pcmlow, **fftopts)
    Pxx /= np.max(Pxx)
    
    
    # Perform matched filtering with complex IQ data
    # TODO: I can probably downsample by a factor of 4 (from 44 KHz to 11 KHz)
    # this doesn't seem to work yet
    decimate = 1
    kernel_len = fs // bitrate
    tkern = np.arange(0.0, kernel_len/bitrate, 1.0/bitrate)
    y1 = np.exp(2*np.pi*1j*(f1-fc)*tkern)
    corr_f1 = np.abs(signal.correlate(pcmiq[::decimate], y1[::decimate], mode='same'))
    # Can we figure out if there's a frequency offset?
    y2 = np.exp(2*np.pi*1j*(f2-fc)*tkern)
    corr_f2 = np.abs(signal.correlate(pcmiq[::decimate], y2[::decimate], mode='same'))
    corr_f1 /= np.max(corr_f1)
    corr_f2 /= np.max(corr_f2)
    corr = corr_f2 - corr_f1


    fcorr = interpolate.interp1d(t1[::decimate], corr, fill_value=0.0, bounds_error=False)

    # edge-filtered
    corr_d = schmitt_trigger(corr, fast=True)

    # Extract bit transition times from here

    # Consider edges within 1/4 symbol of each other as
    # false spurious edges
    mindist = int(np.round(fs / decimate * (0.25 / bitrate)))
    logging.debug(f"edge max {mindist:0.6f}")

    edges_idx = zerocrossings(corr_d, mindist=mindist)
    logging.info(f"Found {len(edges_idx):d} symbol edges")
    # snap edge times to nearest actual time
    # Get the times of each edge as seen by the receiver
    edges_time_rx = [idx * decimate * dt + tstart for idx in edges_idx]
    logging.debug(f"edge max {np.max(edges_time_rx):0.2f}")
    # Get delta timestamp to previous edge
    edges_dtime_rx = np.diff(edges_time_rx, prepend=edges_time_rx[0])
    # Round delta times to nearest multiple of the bitrate, which
    # we assume is the actual time sent on the transmitter timebase
    edges_dtime_tx = np.round(edges_dtime_rx * bitrate) / bitrate
    # Accumulate to get time on transmitter timebase
    # round gets rid of double precision float imprecision
    edges_dtime_tx[0] += tstart
    edges_time_tx = np.round(np.cumsum(edges_dtime_tx), decimals=8)
    


    edges_time_rx2 = np.hstack((0.0, tstart, edges_time_rx, sig_endtime))
    edges_time_tx2 = np.hstack((0.0, tstart, edges_time_tx, sig_endtime))


    # Interpolate to get time on receiver timebase to sample the
    # middle of a bit time
    fsampletime = interpolate.interp1d(edges_time_tx2, edges_time_rx2, kind='linear', copy=False)

    nbits = int(sig_time * bitrate)
    tbits2 = np.arange(tstart, tstart + nbits / bitrate, 1/bitrate)
    tbits2 = fsampletime(tbits2) + (1 / (bitrate*2))
    bits2 = fcorr(tbits2)

    # Convert bits analog waveform to actual array of bits. 
    #bits_d = [1 if x > 0 else 0 for x in bits2]
    bits_d = [0 if x > 0 else 1 for x in bits2]

    # For QC, calculate how far things are from 0
    # a higher q factor means decoding is better
    qfactor = np.sqrt(np.mean((bits2*2) ** 2))
    logging.info(f"demodulation quality: {qfactor:0.3f} / 1.000")


    # Calculate presence of active signal at bit times.
    active_db2 = f_active(t1)
    data_db2 = f_datasq(t1)


    return list(tbits2), bits_d, list(data_db2), #list(active_db2)



###################################################################################
#                         SIGNAL PROCESSING FUNCTIONS                             #
###################################################################################
    
    

def schmitt_trigger(wfm, fast=False):
    """ Transition debouncing
    https://en.wikipedia.org/wiki/Schmitt_trigger
    """
    if fast:
        return np.sign(wfm) * 0.5

    wfm2 = np.empty(wfm.shape)
    a, b = 10.0, 0.1 # amplifier gain, feedback transfer function
    xout = wfm[0] # output node
    for ii, x in enumerate(wfm):
        xout = min(0.5, max(-0.5, xout * b + x * a))
        wfm2[ii] = xout
    return wfm2



def signal_levels(pcm, fs, fprof, minratio=2.0):
    """ 
    Determine whether a signal has power by examining
    the ratio of powers in relevant bands.

    We define a quiet band, digital data band, and profile active band

    # TODO: use periodogram to minimize memory usage and make this
    able to be streamed

    """

    logging.debug(f"Making spectrogram, fs={fs:0.1f}")
    nfft = 4096
    nperseg =  nfft // 2
    f, t, Sxx = signal.spectrogram(pcm, fs=fs, nfft=nfft,
                nperseg=nperseg, noverlap=0, scaling='spectrum', mode='magnitude')
    dt = np.mean(np.diff(t))
    logging.info("Spectrogram size: {:s}, dt={:0.4f} f=[{:0.2f},{:0.2f})"
                "".format(str(Sxx.shape), dt, min(f), max(f) ))

    # Get the indices that represent each band, using bw of about 700 hz
    bw2 = 400.0
    band_fc = [600,         # digital data band
               11000 - 2*bw2, # quiet band
               7500         # profile active
              ]
    # Power levels in each band
    levels = []
    for fc in band_fc:
        flo, fhi = fc - bw2, fc + bw2
        bandidx = np.nonzero((flo <= f) * (f < fhi))
        level = np.sum(Sxx[bandidx[0], :], axis=0)
        levels.append(level)
    # When we use mode='magnitude', scale by 20 instead of 10 for power
    # Signal level of data band
    data_db = 20*np.log10(levels[0] / levels[1])
    data_db -= np.min(data_db)
    # signal level of the profile active tone
    active_db = 20*np.log10(levels[2] / levels[1])
    active_db -= np.min(active_db)
    return t, data_db, active_db




###################################################################################
#                         ZERO CROSSING IDENTIFICATION                            #
###################################################################################

def zerocrossings(data, mindist=100):
    """ Get all zero crossings, but filter out consecutive crossings
    that are within mindist samples of previous crossings.
    TODO: sub-sample interpolation of zero crossings?
    """

    # <= 0 so that if it lands on zero, it isn't missed.
    # the 2nd one will be thrown out by the min distance parameter
    idx_signchange = np.nonzero(data[:-1] * data[1:] <= 0)
    #logging.info("signchange shape: " + str(idx_signchange[0].shape))
    prevchange = np.diff(idx_signchange[0], prepend=idx_signchange[0][0]-mindist)
    # Keep only changes that are far enough apart from previous sign change
    idx_keep = np.nonzero(prevchange >= mindist)
    logging.info(f"Initial crossings: {len(idx_signchange[0]):d} filtered: {len(idx_keep[0]):d}")
    return idx_signchange[0][idx_keep]




    
    


