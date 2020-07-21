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






def demodulateAXCTD(pcmin, fs):
    # TODO: we're cheating by starting after carrier received
    tstart = 0.01
    istart = int(fs * tstart)
    iend = len(pcmin) // 8 #102400


    #basic configuration
    dt = 1/fs
    f1 = 400 # bit 1 (mark) = 400 Hz
    f2 = 800 # bit 0 (space) = 800 Hz
    bitrate = 800 #symbol rate = 800 Hz
    fprof = 7500 #7500 Hz signal associated with active profile data transmission
    #pctchecksig = 0.95 #percent of each segment to check

    #demodulation parameter calculation
    #pointsperbit = int(np.round(fs/bitrate)) #number of PCM datapoints in each bit
    #window = int(np.round(pctchecksig*pointsperbit))
    #tt = np.arange(window)/fs


    # Normalize amplitude of audio signal
    pcm = pcmin[istart:iend] - np.mean(pcmin[istart:iend])
    pcm *= 1.0 / np.max(np.abs(pcm))
    logging.info("PCM signal length: {:d} samples, {:0.3f} seconds"
                 "".format(len(pcm), len(pcm) / fs))


    t1 = np.arange(0, len(pcm) / fs, dt) + tstart

    # make 1200 Hz lowpass filter to separate pilot tone from digital data
    sos = signal.butter(6, 1200, btype='lowpass', fs=fs, output='sos')
    pcmlow = signal.sosfilt(sos, pcm)


    #--------------------------------------------------------------
    logging.debug("Making spectrogram")
    fig2, axs2 = plt.subplots(3, 1, sharex=True)
    nfft = 8192
    nperseg =  nfft // 2
    noverlap = 0
    f, t, Sxx = signal.spectrogram(pcmlow, fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap) #fs // bitrate)
    logging.info("Spectrogram size: " + str(Sxx.shape))

    for ii, nfreqs in enumerate((256, len(f) // 2)):
        axs2[ii].pcolormesh(t + tstart, f[0:nfreqs], 10*np.log10(Sxx[0:nfreqs, :]), shading='gouraud')
        axs2[ii].set_ylabel('Frequency [Hz]')
        axs2[ii].set_xlabel('Time [sec]')


    t2, data_db, active_db = signal_levels(pcm, fs=fs)
    t2 += tstart
    axs2[2].plot(t2, data_db, label='Data')
    axs2[2].plot(t2, active_db, label='Active')
    axs2[2].set_title('Signal Levels')
    axs2[2].set_ylabel('dB')
    axs2[2].set_xlabel('Time [sec]')
    axs2[2].grid(True)
    axs2[2].legend()
    plt.tight_layout()



    fig1, axs = plt.subplots(2, 1, sharex=True)

    # Squelch digital data and display
    f_datasq = interpolate.interp1d(t2, data_db > 10, bounds_error=False, fill_value=0)

    axs[0].plot(t1, pcmlow * f_datasq(t1))
    axs[0].set_title('Digital data (squelched)')
    axs[0].grid(True)


    logging.debug("Done filtering")
    plt.tight_layout()

    # Investigate complex IQ demodulation
    fig3, axs3 = plt.subplots(3, 1, sharex=True)
    fc = 600
    # convert original lowpassed signal to complex domain IQ, centered atfc

    f600hz = np.exp(2*np.pi*1j*fc*t1) * f_datasq(t1)
    sos400 = signal.butter(6, 400, btype='lowpass', fs=fs, output='sos')
    pcmiq = signal.sosfilt(sos400, f600hz * pcmlow)
    pcmiq /= np.max(np.abs(pcmiq))

    # Perform matched filtering with complex IQ data
    #y = exp(2*pi*j*fspac.*t)';
    kernel_len = fs // bitrate
    tkern = np.arange(kernel_len) / bitrate
    y1 = np.exp(2*np.pi*1j*(f1-fc)*tkern)
    corr_f1 = np.abs(signal.correlate(pcmiq, y1, mode='same'))
    y2 = np.exp(2*np.pi*1j*(f2-fc)*tkern)
    corr_f2 = np.abs(signal.correlate(pcmiq, y2, mode='same'))
    corr_f1 /= np.max(np.abs(corr_f1))
    corr_f2 /= np.max(np.abs(corr_f2))

    corr = corr_f1 - corr_f2
    # bit times
    #tbits = np.arange(len(pcm) // (fs / bitrate)) * (1 / bitrate) + tstart
    fcorr = interpolate.interp1d(t1, corr, fill_value=0.0, bounds_error=False)

    args = (fcorr, tstart, fs, len(corr), bitrate)
    tbits0, bits0 = sample_bits0(0.0, *args)

    markeropts = {'marker':'x', 'linewidth': 0, 'markersize': 1.0}

    axs3[0].plot(t1, corr, label='dcorrelation')
    axs3[0].plot(tbits0, bits0, **markeropts)
    axs3[0].grid(True)
    #axs2[1].legend()
    axs3[0].set_xlabel('Time [sec]')
    axs3[0].set_title('Sample Matched Filter Waveform Naively')

    # lowpass filter?

    corr_d = schmitt_trigger(corr)
    fcorr_d = interpolate.interp1d(t1, corr_d)
    args = (fcorr_d, tstart, fs, len(corr), bitrate)
    tbits, bits = sample_bits0(0.0, *args)

    axs3[1].plot(t1, corr_d, label='dcorr_d')
    axs3[1].plot(tbits, bits, **markeropts)
    axs3[1].grid(True)
    #axs2[1].legend()
    axs3[1].set_title('Apply Schmitt Trigger Edge Filter')
    axs3[1].set_xlabel('Time [sec]')

    # Extract bit transition times from here

    diff_corr_d = np.diff(corr_d)
    #edges_idx = np.argwhere(np.abs(diff_corr_d) > 0.5)

    # Consider edges within 1/4 symbol of each other as
    # false spurious edges
    mindist = int(np.round(fs * (0.25 / bitrate)))
    edges_idx = zerocrossings(corr_d, mindist=mindist)
    # snap edge times to nearest actual time
    # (this doesn't seem like it will work)
    tin = [0.0, tstart]
    tout = [0.0, tstart]

    # Get the times of each edge
    edges_times = [float(idx / fs) + tstart for idx in edges_idx]
    # Get delta timestamp to previous edge
    edges_dtimes = np.diff(edges_times, prepend=edges_times[0])
    # Round delta times to nearest multiple of the bitrate
    edges_dtimes_r = np.round(edges_dtimes * bitrate) / bitrate
    # Sum to get rounded actual time
    edges_times_r = np.round(np.cumsum(edges_dtimes_r) + tstart, decimals=8)

    #for ii, (tedge, tedge2) in enumerate(zip(edges_times, edges_times_r)):
    #    logging.info(f"{ii} {tedge:0.6f} {tedge2}")


    time1 = len(pcm) / fs + tstart
    tin = [0.0, tstart] + list(edges_times) + [time1,]
    tout = [0.0, tstart] + list(edges_times_r) + [time1,]

    # Function to convert actual time to desired sample time
    #print(tin, tout)
    logging.info(f"Found {len(edges_idx):d} zero crossings")

    fsampletime = interpolate.interp1d(np.array(tout), np.array(tin), kind='linear')



    nbits = len(pcm) // (fs / bitrate)
    tbits2 = np.arange(nbits) * (1 / bitrate) + tstart
    print(min(tbits), max(tbits2), len(tbits2))
    tbits2 = fsampletime(tbits2) + (1 / (bitrate*2))
    print(min(tbits2), max(tbits2), len(tbits2))
    bits2 = fcorr(tbits2)
    


    axs3[2].plot(t1, corr, label='MF')
    axs3[2].plot(tbits0, bits0, label='naive', **markeropts)
    markeropts['marker'] = '+'
    axs3[2].plot(tbits2, bits2, label='recovered', **markeropts)
    axs3[2].legend()
    axs3[2].grid(True)
    axs3[2].set_title('Sample Matched Filter Waveform with Recovered Symbol Times')

    plt.tight_layout()


    # Convert bits to actual array of bits
    bits_d = bits > 0


    #recover_timing(pcm, t1, fs)

    plt.show()

    

    return None, None, None


def schmitt_trigger(wfm):
    """ TODO: actually turn this into a real hysteresis
    https://en.wikipedia.org/wiki/Schmitt_trigger
    """
    wfm2 = np.empty(wfm.shape)
    a, b = 10.0, 0.0 # amplifier gain, feedback transfer function
    xout = 0.0 # output node
    for ii, x in enumerate(wfm):
        xout = min(0.5, max(-0.5, xout * b + x * a))
        wfm2[ii] = xout

    return wfm2
    #y = np.sign(wfm)
    #return y



def signal_levels(pcm, fs, minratio=2.0):
    """ 
    Determine whether a signal has power by examining
    the ratio of powers in relevant bands.

    We define a quiet band, digital data band, and profile active band

    # TODO: use periodogram to minimize memory usage and make this
    able to be streamed

    """

    logging.debug("Making spectrogram")
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
               11000 - bw2, # quiet band
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
    # signal level of the profile active tone
    active_db = 20*np.log10(levels[2] / levels[1])

    return t, data_db, active_db





def recover_timing(pcm, t1, fs):
    """ Return a function that maps receiver time to transmitter time
    based on waveform
    """

    # make 7000-8000 bandpass filter for pilot tone
    sos = signal.butter(6, [100, 1200], btype='bandpass', fs=fs, output='sos')
    pcm2 = signal.sosfilt(sos, pcm)
    pcm2 /= np.max(np.abs(pcm2))
    fig1, axs1 = plt.subplots(2, 1, sharex=True)

    plt.plot(t1, pcm2)

    return fig1, axs1


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

def sampling_uncertainty(freqoffset, mf_func, tstart, fs, nsamples, bitrate):
    """ Measures the uncertainty of the bit sampling,
    defined as the distance from +-1 of bits.

    i.e., bits that are closer to 0 are more uncertain
    """
    _, bits = sample_bits0(freqoffset, mf_func, tstart, fs, nsamples, bitrate)
    #d = np.mean(np.power(1.0 - np.abs(bits), 4))
    d = np.mean(1.0 - np.abs(bits))
    return d




def sample_bits0(freqoffset, mf_func, tstart, fs, nsamples, bitrate):
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
    tbits = np.arange(nbits) * (1 / bitrate1) + tstart
    bits = mf_func(tbits)
    return tbits, bits




def sample_bits0(freqoffset, mf_func, tstart, fs, nsamples, bitrate):
    """
    Perform symbol timing recovery

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
    tbits = np.arange(nbits) * (1 / bitrate1) + tstart
    bits = mf_func(tbits)
    return tbits, bits

###################################################################################
#                         ZERO CROSSING IDENTIFICATION                            #
###################################################################################

def gen_edges(data, i0):
    """ Generate zero crossings """
    pass

def zerocrossings(data, mindist=100):
    """ Get all zero crossings, but filter out consecutive crossings
    that are within mindist samples of previous crossings.
    """


    # <= 0 so that if it lands on zero, it isn't missed.
    # the 2nd one will be thrown out by the min distance parameter
    idx_signchange = np.nonzero(data[:-1] * data[1:] <= 0)
    logging.info("signchange shape: " + str(idx_signchange[0].shape))
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
    
    
    
    
