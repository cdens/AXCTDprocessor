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


def readAXCTDwavfile(inputfile):
    
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
    
    logging.info(f"Demodulating audio stream (size {np.shape(audiostream)})")
    
    return audiostream, fs





    


def demodulate_axctd(pcmin, fs, timerange):
    """ 
    Note for the future: We could probably adapt the code here for a realtime algorithm:
    https://github.com/EliasOenal/multimon-ng/blob/master/demod_afsk12.c
    """

    # Change these variables to allow partial file processing for debugging
    tstart = timerange[0]
    istart = int(fs * tstart) #AXCTD profile starts at 13:43 (length=21:43) therefore 63% of the way in
    
    #how much of audio file to select
    if timerange[1] == -1:
        iend = len(pcmin)
    else:
        iend = int(fs*timerange[1]) 
        
    #basic configuration
    f1 = 400 # bit 1 (mark) = 400 Hz
    f2 = 800 # bit 0 (space) = 800 Hz
    bitrate = 800 #symbol rate = 800 Hz
    fprof = 7500 #7500 Hz signal associated with active profile data transmission
    squelch_snr = 10.0 # db

    # Basic signal accounting information
    dt = 1/fs
    len_pcm = iend - istart # number of samples
    sig_time = len_pcm * dt # signal time duration
    sig_endtime = tstart + sig_time
    
    if istart >= iend:
        logging.info(f"[!] Start index is after end index!\nStart={istart}, End={iend}")
        exit()
    
    # Normalize amplitude/DC offset of audio signal
    pcm_dc = np.mean(pcmin[istart:iend])
    pcm_ampl = np.max(np.abs(pcmin[istart:iend]))
    pcm = (pcmin[istart:iend].astype(np.float) - pcm_dc) / pcm_ampl
    logging.info("PCM signal length: {:d} samples, {:0.3f} seconds"
                 "".format(len_pcm, sig_time))

    while fs >= 44100: # downsample
        pcm = signal.decimate(pcm, 2)
        fs /= 2
        # Basic signal accounting information
        dt = 1/fs
        len_pcm = len(pcm)
        sig_time = len_pcm * dt # signal time duration
        sig_endtime = tstart + sig_time


    t1 = np.arange(tstart, sig_endtime, dt)
    
    #correcting length issues
    if len(t1) > len(pcm):
        t1 = t1[:len(pcm)]
    elif len(pcm) > len(t1): #this hasn't happened but just in case
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


    return list(tbits2), bits_d, list(data_db2), list(active_db2)



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


    

