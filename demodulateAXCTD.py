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

def demodulateAXCTD0(pcm, fs):
    
    #basic configuration
    dt = 1/fs
    f1 = 400 #bit 1 = 400 Hz
    f2 = 800 #bit 0 = 800 Hz
    bitrate = 800 #symbol rate = 800 Hz
    fprof = 7500 #7500 Hz signal associated with active profile data transmission
    pctchecksig = 0.95 #percent of each segment to check
    
    #demodulation parameter calculation
    pointsperbit = int(np.round(fs/bitrate)) #number of PCM datapoints in each bit
    window = int(np.round(pctchecksig*pointsperbit))
    tt = np.arange(window)/fs
        
    #test signals for individual point FT
    y1 = np.cos(2*np.pi*f1*tt) + 1j*np.sin(2*np.pi*f1*tt)
    y2 = np.cos(2*np.pi*f2*tt) + 1j*np.sin(2*np.pi*f2*tt)
    yprof = np.cos(2*np.pi*fprof*tt) + 1j*np.sin(2*np.pi*fprof*tt)
    
    #box-smoothed PCM signal to identify/track zero crossings
    pcmlow = boxSmooth(pcm, int(np.round(3*fs/fprof)))
    
    #initializing loop
    isDone = False
    siglen = len(pcm)
    bitnum = np.floor(siglen/pointsperbit)
    start = 0 #start checking data at first point
    p7500 = []
    bitstream = []
    times = []
    c = 0
    
    #loop until end of PCM data
    while not isDone:
        
        if start < siglen - 2*pointsperbit: #won't overrun PCM data- stops ~2 bits short 
        
            #time
            times.append(start/fs)    
        
            #calculating power at each frequency
            cpcm = pcm[start:start+window]
            S1 = np.abs(np.trapz(y1*cpcm, dx=dt))
            S2 = np.abs(np.trapz(y2*cpcm, dx=dt))
            p7500.append(np.abs(np.trapz(yprof*cpcm, dx=dt)))
            
            #appending bit to stream based on which is more powerful
            if S1 > S2:
                cbit = 1
            else:
                cbit = 0
            bitstream.append(cbit)
                
            #look for upcrossings/downcrossings in expected ranges
            bitranges = [[0.9,1.1],[0.8,1.2],[0.7,1.3],[0.4,1.6]]

            for br in bitranges:
                upcoming = pcmlow[start:start+int(np.round(br[1]*pointsperbit))]
                ind = getnextncrossings(upcoming, 2-cbit, int(np.round(br[0]*pointsperbit)))
                
                if ind:
                    break
            
            #increase start point
            if not ind:
                ind = pointsperbit
            start += ind
            
            c += 1
            if c%1E4 == 0:
                print(f"\r[+] Demodulating file, progress: {int(100*c/bitnum):2d}%",end=" ")
                
        else:
            isDone = True
            
    
    #convert 7500 Hz signals to ndarray, smooth
    p7500 = boxSmooth(np.asarray(p7500), 50)
    print("\r[+] Demodulating file, progress: 100%")
    
    return bitstream, times, p7500




def demodulate_axctd(pcmin, fs, plot=False):

    # Change these variables to allow partial file processing for debugging
    tstart = 0.01
    istart = int(fs * tstart)
    iend = len(pcmin)


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

    assert istart < iend
    # Normalize amplitude/DC offset of audio signal
    pcm_dc = np.mean(pcmin[istart:iend])
    pcm_ampl = np.max(np.abs(pcmin[istart:iend]))
    pcm = (pcmin[istart:iend] - pcm_dc) / pcm_ampl
    logging.info("PCM signal length: {:d} samples, {:0.3f} seconds"
                 "".format(len_pcm, sig_time))

    if fs >= 44100: # downsample
        pcm = signal.decimate(pcm, 2)
        fs /= 2
        # Basic signal accounting information
        dt = 1/fs
        len_pcm = len(pcm)
        sig_time = len_pcm * dt # signal time duration
        sig_endtime = tstart + sig_time


    t1 = np.arange(tstart, sig_endtime, dt)
    # TODO: make this filter configurable
    # Use bandpass filter to extract digital data only
    #sos = signal.butter(6, [f1 * 0.5, f2 * 1.5], btype='bandpass', fs=fs, output='sos')
    #sos = signal.butter(6, [5, 1200], btype='bandpass', fs=fs, output='sos')
    sos = signal.butter(6, 1200, btype='lowpass', fs=fs, output='sos')
    #sos = signal.cheby1(6, 0.1, 1200, btype='lowpass', fs=fs, output='sos')
    pcmlow = signal.sosfilt(sos, pcm)

    figures = []
    #--------------------------------------------------------------
    logging.debug("Making spectrogram")

    if plot:
        fig2, axs2 = plt.subplots(3, 1, sharex=True)
        figures.append(fig2)
        nfft = 8192
        nperseg =  nfft // 2
        noverlap = 0
        f, t, Sxx = signal.spectrogram(pcmlow, fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap) #fs // bitrate)
        logging.debug("Spectrogram size: " + str(Sxx.shape))

        for ii, nfreqs in enumerate((256, len(f) // 2)):
            axs2[ii].pcolormesh(t + tstart, f[0:nfreqs], 10*np.log10(Sxx[0:nfreqs, :]), shading='gouraud')
            axs2[ii].set_ylabel('Frequency [Hz]')
            axs2[ii].set_xlabel('Time [sec]')


    t2, data_db, active_db = signal_levels(pcm, fs=fs, fprof=fprof)
    t2 += tstart

    if plot:
        axs2[2].plot(t2, data_db, label='Data')
        axs2[2].plot(t2, active_db, label='Active')
        axs2[2].set_title('Signal Levels')
        axs2[2].set_ylabel('dB')
        axs2[2].set_xlabel('Time [sec]')
        axs2[2].grid(True)
        axs2[2].legend()
        plt.tight_layout()


    # Squelch digital data and display
    f_datasq = interpolate.interp1d(t2, data_db, bounds_error=False, fill_value=0)
    f_active = interpolate.interp1d(t2, active_db, bounds_error=False, fill_value=0)

    squelch_mask_t1 = (f_datasq(t1) > squelch_snr)
    pcm_sq = pcmlow * squelch_mask_t1

    if plot:
        fig1, axs = plt.subplots(2, 1, sharex=True)
        figures.append(fig1)

        axs[0].plot(t1, pcm_sq)
        axs[0].set_title('Digital data (squelched)')
        axs[0].grid(True)

        axs[1].plot(t1, f_datasq(t1), label='Data Signal')
        axs[1].plot(t1, 50*squelch_mask_t1, label='Squelch Mask')
        plt.tight_layout()

    logging.debug("Done filtering")
    # Perform complex IQ downconversion, demodulation

    fc = (f1 + f2) / 2
    # convert original lowpassed signal to complex domain IQ, centered at fc

    # downconvert and lowpass filter
    fcarrier = np.exp(2*np.pi*1j*-fc*t1)
    # TODO: make the downconversion filter frequency configurable
    sosdcx = signal.butter(4, fc*0.6, btype='lowpass', fs=fs, output='sos')
    pcmiq = signal.sosfilt(sosdcx, fcarrier * pcmlow)
    pcmiq /= np.max(np.abs(pcmiq))
    pcmiq *= squelch_mask_t1





    fftopts = {'fs': fs, 'nfft': 16384, 'scaling': 'spectrum', 'return_onesided': False}

    f, Pxx = signal.periodogram(pcmlow, **fftopts)
    Pxx /= np.max(Pxx)
    #print(max(pcm_sq), min(pcm_sq))
    #print(f)
    #print(Pxx)
    #print(np.max(np.abs(Pxx)))
    if plot:
        fig4, axs4 = plt.subplots(3, 1)
        figures.append(fig4)


        dbeps = 1e-9
        # Show unfiltered PSD of original using periodogram
        half = len(f) // 2
        axs4[1].plot(f, 10*np.log10(Pxx + dbeps), label='pcmlow')
        axs4[1].set_xlabel('Frequency [Hz]')
        axs4[1].set_ylabel('PSD [dB/Hz]')
        axs4[1].grid(True)

        # frequencies for filter
        fh = np.linspace(-fs/4, fs/4, fftopts['nfft'])

        # Show unfiltered PSD of downconverted using periodogram
        f, Pxx = signal.periodogram(fcarrier*pcmlow, **fftopts)
        Pxx /= np.max(Pxx)
        axs4[1].plot(f, 10*np.log10(Pxx + dbeps), label='dcx signal')
        axs4[1].set_xlabel('Frequency [Hz]')
        axs4[1].set_ylabel('PSD [dB/Hz]')
        axs4[1].grid(True)

        # overlay low pass filter
        w, h = signal.sosfreqz(sos, fs=fs, worN=fh)
        axs4[1].plot(w, 20 * np.log10(np.abs(h) + dbeps), label='aa lpf')
        axs4[1].legend()
        #axs4[2].set_title('Butterworth filter frequency response')
        #axs4[2].set_xlabel('Frequency [Hz]')
        #axs4[2].set_ylabel('Amplitude [dB]')
        #axs4[2].margins(0, 0.1)
        #axs4[2].grid(which='both', axis='both')
        #axs4[1].axvline(1200, color='green') # cutoff frequency
        plt.tight_layout()

        # overlay low pass filter
        pcmiq2 = signal.sosfilt(sosdcx, fcarrier * pcmlow)
        f, Pxx = signal.periodogram(np.real(pcmiq2), **fftopts)
        Pxx /= np.max(Pxx)
        axs4[2].plot(f, 10*np.log10(Pxx + dbeps), label='dcx data')
        w, h = signal.sosfreqz(sosdcx, fs=fs, worN=fh)
        axs4[2].plot(w, 20 * np.log10(np.abs(h) + dbeps), label='dcx lpf')
        axs4[2].set_title('Downconversion')
        axs4[2].set_xlabel('Frequency [Hz]')
        axs4[2].set_ylabel('Amplitude [dB]')
        axs4[2].margins(0, 0.1)
        axs4[2].grid(which='both', axis='both')
        axs4[2].legend()
        plt.tight_layout()


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

    if plot:
        args = (fcorr, tstart, fs, len_pcm, bitrate)
        tbits0, bits0 = sample_bits(0.0, *args)

        lineopts = {'linewidth': 0.75}
        markeropts = {'marker':'x', 'linewidth': 0}

        fig3, axs3 = plt.subplots(3, 1, sharex=True)
        figures.append(fig3)
        axs3[0].plot(t1[::decimate], corr, label='dcorrelation', **lineopts)
        axs3[0].plot(tbits0, bits0, **markeropts)
        axs3[0].grid(True)
        #axs2[1].legend()
        axs3[0].set_xlabel('Time [sec]')
        axs3[0].set_title('Sample Matched Filter Waveform Naively')


    # edge-filtered
    corr_d = schmitt_trigger(corr, fast=True)

    if plot:
        fcorr_d = interpolate.interp1d(t1[::decimate], corr_d, kind='linear', copy=False)
        args = (fcorr_d, tstart, fs, len_pcm, bitrate)
        tbits, bits = sample_bits(0.0, *args)
        axs3[1].plot(t1[::decimate], corr_d, label='dcorr_d', **lineopts)
        axs3[1].plot(tbits, bits, **markeropts)
        axs3[1].grid(True)
        axs3[1].set_title('Apply Schmitt Trigger Edge Filter')
        axs3[1].set_xlabel('Time [sec]')

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

    #logging.debug(f"edges_time_tx {edges_time_tx[0]:0.6f}")
    #logging.debug(f"edges_time_tx {edges_time_tx[len(edges_time_tx)//2]:0.6f}")
    #logging.debug(f"edges_time_tx {edges_time_tx[-1]:0.6f}")


    edges_time_rx2 = np.hstack((0.0, tstart, edges_time_rx, sig_endtime))
    edges_time_tx2 = np.hstack((0.0, tstart, edges_time_tx, sig_endtime))


    # Interpolate to get time on receiver timebase to sample the
    # middle of a bit time
    fsampletime = interpolate.interp1d(edges_time_tx2, edges_time_rx2, kind='linear', copy=False)

    nbits = int(sig_time * bitrate)
    tbits2 = np.arange(tstart, tstart + nbits / bitrate, 1/bitrate)
    tbits2 = fsampletime(tbits2) + (1 / (bitrate*2))
    bits2 = fcorr(tbits2)

    if plot:
        axs3[2].plot(t1[::decimate], corr, label='MF', **lineopts)
        axs3[2].plot(tbits0, bits0, label='naive', **markeropts)
        markeropts['marker'] = '+'
        axs3[2].plot(tbits2, bits2, label='recovered', **markeropts)
        axs3[2].legend()
        axs3[2].grid(True)
        axs3[2].set_title('Sample Matched Filter Waveform with Recovered Symbol Times')

        plt.tight_layout()

    # Convert bits analog waveform to actual array of bits for a bitstring
    bits_d = ''.join(['1' if x > 0 else '0' for x in bits2])

    # For QC, calculate how far things are from 0
    # a higher q factor means decoding is better
    qfactor = np.sqrt(np.mean((bits2*2) ** 2))
    logging.info(f"demodulation quality: {qfactor:0.3f} / 1.000")

    if plot:
        # Demodulation quality
        #fig4, axs4 = plt.subplots(2, 1)
        #figures.append(fig4)

        has_sig = np.nonzero(f_datasq(tbits2) > squelch_snr)

        hsig, bin_edges = np.histogram(bits2[has_sig], bins=100)
        x = (bin_edges[0:-1] + np.diff(bin_edges) / 2)
        #f = (x + 0.5) * (f2 - f1) + f1
        axs4[0].plot(x*2, hsig)
        axs4[0].set_xlabel('Matched Filter Output')
        axs4[0].set_ylabel('Count')
        axs4[0].grid(True)
        axs4[0].set_title('Demodulation Quality')
        plt.tight_layout()


    # Calculate presence of active signal at bit times.
    active_db2 = f_active(t1)
    data_db2 = f_datasq(t1)

    if plot:
        pass #fig3.clear()

    return list(tbits2), bits_d, list(data_db2), list(active_db2), figures


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







def sample_bits(freqoffset, mf_func, tstart, fs, nsamples, bitrate):
    """ 
    Sample the bits out of an array at the given bitrate.

    Given a matched filter decision array with values between 1 and 0
    sample the array at the given sampling parameters
    mf_func is an interpolating function based on a matched filter output
    fs is sampling rate in Hz of the input waveform
    tstart is the time to start sampling in the input waveform
    nsamples is number of samples in the input waveform
    bitrate is the number of bits per second to output
    freqoffset is the frequency offset in Hz between the transmitter and
    receiver. a value > 0 means the oscillator of the transmitter system
    is biased faster than the oscillator of the receiver system
    """
    bitrate1 = bitrate + freqoffset
    nbits = nsamples // (fs / bitrate1)
    tbits = np.arange(tstart, nbits / bitrate1 + tstart, 1/bitrate1)
    bits = mf_func(tbits)
    return tbits, bits




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




def getnextncrossings(data, ncrossings, minind):
    
    ind = 0 #if checks fail- return no zero crossings found
    
    try:
        
        
        #getting indices corresponding to all upcrossings/downcrossings
        allcrossings = np.sort(np.concatenate((np.where((data[:-1] < 0) & (data[1:] >= 0))[0], 
            np.where((data[:-1] >= 0) & (data[1:] < 0))[0] )))
        
        #removing all jumps back and forth across 0 within 6 points of one another
        allcrossings = np.delete(allcrossings, np.diff(allcrossings,append=allcrossings[-1]+10) <= 6)
        
        #1 crossing expected: identifying first zero crossing within valid range
        if ncrossings == 1:
            allcrossings = allcrossings[allcrossings >= minind] #within range
            ind = allcrossings[0]
        
        #2 crossings expected: getting second valid crossing, must be within range and first crossing must be within 0.5*range    
        elif ncrossings == 2:
            allcrossings = allcrossings[allcrossings >= minind/2] #first crossing within 0.5*range
            allcrossings = [c for (i,c) in enumerate(allcrossings) if (i >= 2 and c >= minind)] #get second crossing (**now list not ndarray)
            ind = allcrossings[0]
    
    except IndexError:
        pass
    
    return ind
            

    
###################################################################################
#                                   BOX SMOOTHER                                  #
###################################################################################
    

def boxSmooth(y, npts):
    box = np.ones(npts)/npts
    return np.convolve(y, box, mode='same')
    
    

