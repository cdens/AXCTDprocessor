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
import matplotlib.pyplot as plt



###################################################################################
#                         AXCTD FSK DEMODULATION ROUTINE                          #
###################################################################################
       
def demodulate(pcmin, fs, edge_buffer, high_bit_scale):
        
    #basic configuration
    f1 = 400 # bit 1 (mark) = 400 Hz
    f2 = 800 # bit 0 (space) = 800 Hz
    bitrate = 800 #symbol rate = 800 Hz
    fprof = 7500 #7500 Hz signal associated with active profile data transmission
    squelch_snr = 10.0 # db

    # Basic signal accounting information
    dt = 1/fs
    len_pcm = len(pcmin) # number of samples
    sig_time = len_pcm * dt # signal time duration    
        
    # Normalize amplitude/DC offset of audio signal
    pcm_dc = np.mean(pcmin)
    pcm_ampl = np.max(np.abs(pcmin))
    pcm = (pcmin.astype(np.float) - pcm_dc) / pcm_ampl
    
    #lowpass filtering signal to remove 7500 Hz tone if applicable
    sos = signal.butter(6, 1200, btype='lowpass', fs=fs, output='sos')
    pcmlow = signal.sosfilt(sos, pcm)
    
    #simple estimate of bit edge times assuming no symbol drift
    t1 = np.arange(0, sig_time, dt)
    
    #correcting length issues that may arise
    if len(t1) > len(pcm):
        t1 = t1[:len(pcm)]
    elif len(pcm) > len(t1):
        pcm = pcm[:len(t1)]

    #return signal levels in data and profile tone frequency bands
    time_siglev, data_db, active_db = signal_levels(pcm, fs=fs, fprof=fprof)
    
    #interpolation functions for active and data band signal levels
    f_datasq = interpolate.interp1d(time_siglev, data_db, bounds_error=False, fill_value=0)
    f_active = interpolate.interp1d(time_siglev, active_db, bounds_error=False, fill_value=0)
    
    # Squelch digital data
    squelch_mask_t1 = (f_datasq(t1) > squelch_snr) #logical, true where signal meets minimum SNR requirement

    #carrier/center frequency halfway between mark and space bits
    fc = (f1 + f2) / 2

    # downconvert and lowpass filter complex signal to demodulate
    fcarrier = np.exp(2*np.pi*1j*-fc*t1)
    sosdcx = signal.butter(4, fc*0.6, btype='lowpass', fs=fs, output='sos')
    pcmiq = signal.sosfilt(sosdcx, fcarrier * pcmlow) #Perform complex IQ downconversion, demodulation
    pcmiq /= np.max(np.abs(pcmiq)) #normalize
    pcmiq *= squelch_mask_t1 #drop all PCM data where SNR isn't met
    
    # Perform matched filtering with complex IQ data
    kernel_len = fs // bitrate #number of PCM points per bit
    tkern = np.arange(0.0, kernel_len/bitrate, 1.0/bitrate)
    
    y1 = np.exp(2*np.pi*1j*(f1-fc)*tkern) #complex terms for power/covariance calculation
    y2 = np.exp(2*np.pi*1j*(f2-fc)*tkern)
    corr_f1 = np.abs(signal.correlate(pcmiq, y1, mode='same')) #correlating signals, calculating confidence
    corr_f2 = np.abs(signal.correlate(pcmiq, y2, mode='same'))
    corr_f1 /= np.max(corr_f1)
    corr_f2 /= np.max(corr_f2)
    corr = corr_f2 - corr_f1 # - high_bit_scale #corrects for amplitude difference between mark/space freqs
    
    
    # Transition debouncing
    # https://en.wikipedia.org/wiki/Schmitt_trigger
    corr_d = np.sign(corr) * 0.5 #Schmitt trigger (fast) 
    
    # Extract bit transition times from here
    # Consider edges within 1/4 symbol of each other as
    # false spurious edges (mindist is min distance between zero crossings)
    mindist = int(np.round(fs * (0.25 / bitrate)))
    
    #identifying bit edges
    edges_idx = zerocrossings(corr_d, mindist=mindist)
    
    # snap edge times to nearest actual time
    # Get the times of each edge as seen by the receiver
    edges_time_rx = [idx * dt for idx in edges_idx]
    
    # Get delta timestamp to previous edge
    edges_dtime_rx = np.diff(edges_time_rx, prepend=edges_time_rx[0])
    
    # Round delta times to nearest multiple of the bitrate, which
    # we assume is the actual time sent on the transmitter timebase
    edges_dtime_tx = np.round(edges_dtime_rx * bitrate) / bitrate
    
    # Accumulate to get time on transmitter timebase
    # round gets rid of double precision float imprecision
    edges_time_tx = np.round(np.cumsum(edges_dtime_tx), decimals=8)
    edges_time_rx2 = np.hstack((0.0, 0, edges_time_rx, sig_time))
    edges_time_tx2 = np.hstack((0.0, 0, edges_time_tx, sig_time))
    
    #sampling signal at the middle of each bit to ID bit classification confidence
    nbits = int(sig_time * bitrate)
    bit_times = np.arange(0, nbits / bitrate, 1/bitrate)
    
    # Interpolate bit ties to get time on receiver timebase to sample the
    # middle of a bit time
    fsampletime = interpolate.interp1d(edges_time_tx2, edges_time_rx2, kind='linear', copy=False)
    bit_times = fsampletime(bit_times) + (1 / (bitrate*2))
    
    #interpolate correlation to bit edge times
    fcorr = interpolate.interp1d(t1, corr, fill_value=0.0, bounds_error=False)
    conf = fcorr(bit_times)

    #determine actual array of bits
    curbits = [0 if x > 0 else 1 for x in conf] 
    
    # # Calculate presence of active signal at bit times. #TODO: DELETE THIS
    # active_db2 = f_active(t1)
    # data_db2 = f_datasq(t1)
    
    #calculate approximate bit edge indices from sample times, get index to start demodulation on next pass
    bit_edges = [int(np.round(fs*ct)) for ct in bit_times]
    next_demod_ind = bit_edges[-1] - 1
    
    #remove data up to buffer point
    ci = 0
    while bit_edges[ci] < edge_buffer:
        ci += 1
    bit_edges = bit_edges[ci:]
    curbits = curbits[ci:]
    conf = conf[ci:]
    
    # return list(bit_times), curbits, list(data_db2), list(active_db2)
    
    return curbits, conf, bit_edges, next_demod_ind
    

    
    
    
#Determine whether a signal has power by examining the ratio of powers in relevant bands (quiet, digital data, profile active)
def signal_levels(pcm, fs, fprof, minratio=2.0):

    nfft = 4096
    nperseg =  nfft // 2
    f, t, Sxx = signal.spectrogram(pcm, fs=fs, nfft=nfft,
                nperseg=nperseg, noverlap=0, scaling='spectrum', mode='magnitude')
    dt = np.mean(np.diff(t))

    # Get the indices that represent each band, using bw of about 700 hz
    bw2 = 400.0
    band_fc = [600,         # digital data band
               11000 - 2*bw2, # quiet band
               7500]        # profile active
    
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




    
#Get all zero crossings, but filter out consecutive crossings within mindist samples of previous crossings
def zerocrossings(data, mindist=100):
    
    # <= 0 so that if it lands on zero, it isn't missed.
    # the 2nd one will be thrown out by the min distance parameter
    idx_signchange = np.nonzero(data[:-1] * data[1:] <= 0)
    
    #logging.info("signchange shape: " + str(idx_signchange[0].shape))
    prevchange = np.diff(idx_signchange[0], prepend=idx_signchange[0][0]-mindist)
    
    # Keep only changes that are far enough apart from previous sign change
    idx_keep = np.nonzero(prevchange >= mindist)

    return idx_signchange[0][idx_keep]
    
    
