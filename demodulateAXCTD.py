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
from scipy import signal, interpolate, optimize
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
    tstart = 7.2
    istart = int(fs * tstart)


    #basic configuration
    dt = 1/fs
    f1 = 400 # bit 1 (mark) = 400 Hz
    f2 = 800 # bit 0 (space) = 800 Hz
    bitrate = 800 #symbol rate = 800 Hz
    fprof = 7500 #7500 Hz signal associated with active profile data transmission
    pctchecksig = 0.95 #percent of each segment to check
    
    #demodulation parameter calculation
    pointsperbit = int(np.round(fs/bitrate)) #number of PCM datapoints in each bit
    window = int(np.round(pctchecksig*pointsperbit))
    tt = np.arange(window)/fs

    maxsamples = len(pcmin) // 8 #102400
    # Normalize amplitude of audio signal
    pcm = pcmin[istart:maxsamples] - np.mean(pcmin[istart:maxsamples])
    pcm *= 1.0 / np.max(pcm)
    logging.info("PCM signal length: {:d} samples, {:0.3f} seconds"
                 "".format(len(pcm), len(pcm) / fs))


    t1 = np.arange(0, len(pcm) / fs, dt) + tstart

    fig1, axs = plt.subplots(4, 1, sharex=True)

    
    # make 1200 Hz lowpass filter to separate pilot tone from digital data
    sos = signal.butter(4, 1200, btype='lowpass', fs=fs, output='sos')
    pcmlow = signal.sosfilt(sos, pcm)



    fs2 = 2400
    nsamples = fs // fs2
    #assert fs == 44100
    #q = 20
    #fs2 = fs // q # resample from 44.1 KHz to 44100

    #pcmlow2 = signal.resample(pcmlow, num=nsamples)
    #pcmhi  = pcm - pcmlow

    axs[0].plot(t1, pcmlow)
    axs[0].set_title('Digital data')
    axs[0].grid(True)


    # make highpass filter to act as a squelch.
    # if power is present, squelch.
    sos_sq = signal.butter(2, 10000, btype='highpass', fs=fs, output='sos')
    pcm_sq = signal.sosfilt(sos_sq, pcm)

    



    # make 7000-8000 bandpass filter for pilot tone
    sos = signal.butter(6, [7400, 7600], btype='bandpass', fs=fs, output='sos')
    pcm7500hz = signal.sosfilt(sos, pcm)
    pcm7500hz /= np.max(np.abs(pcm7500hz))



    axs[1].plot(t1, pcm7500hz)
    axs[1].set_title('Pilot tone')
    axs[1].grid(True)
    
    ##############################################
    # Use 400 hz signal as frequency reference
    f400hz = np.exp(2*np.pi*1j*400.0*t1)
    # make 200-600 bandpass filter to track phase of 400 Hz data
    sos = signal.butter(4, [300, 500], btype='bandpass', fs=fs, output='sos')
    pcm400hz = signal.sosfilt(sos, pcm)

    # Calculate the frequency/phase offset of the pilot tone from our internal
    # reference
    soslp = signal.butter(4, 100, btype='lowpass', fs=fs, output='sos')
    fmix = signal.sosfilt(soslp, f400hz * pcm400hz)
    fmix /= np.max(np.abs(fmix))
    #fmix_dphase = np.degrees(np.diff(np.unwrap(np.angle(fmix))))
    # Relative phase between reference and transmitted
    fmix_phase = np.unwrap(np.angle(fmix))

    # Calculate actual times (do we add or subtract?)
    t2 = t1 - fmix_phase / (2*np.pi*400)


    axs[2].plot(t1, np.real(f400hz), label='intref', linewidth=1.0)
    axs[2].plot(t1, 0.7071*pcm400hz, label='TX', linewidth=1.0)

    # Internal ref, timebase adjusted to actual
    axs[2].plot(t2, -0.5*np.real(f400hz), label='intref-adj', linewidth=1.0)
    axs[2].set_title('400 Hz Tone')
    axs[2].grid(True)
    axs[2].legend()


    #axs[3].plot(t1, fmix_phase / (2*np.pi), label='Phase')
    #axs[3].set_ylabel('cycles')
    #axs[3].grid(True)
    #axs[3].set_title('400 Hz Phase Tracking')

    # end 400 hz frequency reference
    #---------------------------------

    # perform autocorrelation

    #pcm400auto = autocorr(pcmlow)
    #axs[3].plot(t1, pcm400auto)
    #axs[3].set_title('autocorrelation')



    logging.debug("Done filtering")
    plt.tight_layout()


    logging.debug("Making spectrogram")
    fig2, axs2 = plt.subplots(4, 1, sharex=True)
    nfft = 8192
    nperseg =  nfft // 2
    noverlap = 0
    f, t, Sxx = signal.spectrogram(pcmlow, fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap) #fs // bitrate)
    logging.info("Spectrogram size: " + str(Sxx.shape))
    #exit()
    nfreqs = 256
    axs2[0].pcolormesh(t + tstart, f[0:nfreqs], 10*np.log10(Sxx[0:nfreqs, :]), shading='gouraud')

    axs2[0].set_ylabel('Frequency [Hz]')
    axs2[0].set_xlabel('Time [sec]')

    # Show matched filters
    #y = exp(2*pi*j*fspac.*t)';
    kernel_len = fs // bitrate
    tkern = np.arange(kernel_len) / bitrate
    y1 = np.exp(2*np.pi*1j*f1*tkern)
    corr_f1 = np.abs(signal.correlate(pcmlow, y1, mode='same'))
    y2 = np.exp(2*np.pi*1j*f2*tkern)
    corr_f2 = np.abs(signal.correlate(pcmlow, y2, mode='same'))
    corr_f1 -= np.mean(corr_f1)
    corr_f2 -= np.mean(corr_f2)
    corr_f1 /= np.max(np.abs(corr_f1))
    corr_f2 /= np.max(np.abs(corr_f2))

    y2 = 0.5*np.sin(2*np.pi*f2*(t1 + 0.25 / f2))
    axs2[1].plot(t1, y2, linewidth=0.75, label='bits')

    #axs2[1].plot(t1, corr_f1 / np.max(corr_f1), label=f'{f1:0.0f}')
    #axs2[1].plot(t1, corr_f2 / np.max(corr_f2), label=f'{f2:0.0f}')
    corr = corr_f1 - corr_f2
    # bit times
    #tbits = np.arange(len(pcm) // (fs / bitrate)) * (1 / bitrate) + tstart
    fcorr = interpolate.interp1d(t1, corr)
    tbits, bits = sample_bits0(0.0, fcorr, tstart, fs, len(pcm), bitrate)
    print(tbits.shape, bits.shape)


    axs2[1].plot(t1, corr, label='dcorrelation')
    axs2[1].plot(tbits, bits, marker='x', linewidth=0)
    axs2[1].grid(True)
    #axs2[1].legend()
    axs2[1].set_xlabel('Time [sec]')


    #---------------------------------------------
    # Synchronize time base
    # internal 7500 hz reference
    f7500hz = np.exp(2*np.pi*1j*7500.0*t1)

    # Calculate the frequency/phase offset of the pilot tone from our internal
    # reference
    sos75 = signal.butter(4, 1000, btype='lowpass', fs=fs, output='sos')
    fmix = signal.sosfilt(sos75, f7500hz * pcm7500hz)
    fmix /= np.max(np.abs(fmix))
    #fmix_dphase = np.degrees(np.diff(np.unwrap(np.angle(fmix))))
    # Relative phase between reference and transmitted
    fmix_phase = np.unwrap(np.angle(fmix))

    # Calculate actual times (do we add or subtract?)
    t2 = t1 + fmix_phase / (2*np.pi*7500)


    axs2[2].plot(t1, np.real(f7500hz), label='intref', linewidth=1.0)

    # Internal ref, timebase adjusted to actual
    axs2[2].plot(t2, -0.7071*np.real(f7500hz), label='intref-adj', linewidth=1.0)


    axs2[2].plot(t1, pcm7500hz, label='TX', linewidth=1.0, alpha=0.75)

    axs2[2].set_title('7.5 KHz pilot tone')



    axs2[2].grid(True)
    axs2[2].legend()


    axs2[3].plot(t1, fmix_phase / (2*np.pi), label='Phase')
    axs2[3].set_ylabel('cycles')
    axs2[3].grid(True)

    


    plt.tight_layout()



    # Investigate complex IQ demodulation
    fig3, axs3 = plt.subplots(3, 1, sharex=True)
    fc = 600
    # convert original lowpassed signal to complex domain IQ, centered atfc

    f600hz = np.exp(2*np.pi*1j*fc*t1)
    sos400 = signal.butter(4, 400, btype='lowpass', fs=fs, output='sos')
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
    fcorr = interpolate.interp1d(t1, corr)

    # Solve for best frequency offset between receiver and transmitter
    args = (fcorr, tstart, fs, len(corr), bitrate)
    logging.info("args: " + str(args))
    #freqopt = optimize.minimize(sampling_uncertainty, args=args,
    #                            x0=0.0, bounds=((-0.5, 0.5),))
    #foffset = float(freqopt.x)
    foffset = 0.0
    logging.info(f"Frequency offset: {foffset:0.3f} Hz")

    tbits, bits = sample_bits0(foffset, *args)
    print(tbits.shape, bits.shape)

    axs3[0].plot(t1, corr, label='dcorrelation')
    axs3[0].plot(tbits, bits, marker='x', linewidth=0)
    axs3[0].grid(True)
    #axs2[1].legend()
    axs3[0].set_xlabel('Time [sec]')

    # lowpass filter?

    corr_d = schmitt_trigger(corr)
    fcorr_d = interpolate.interp1d(t1, corr_d)
    args = (fcorr_d, tstart, fs, len(corr), bitrate)
    tbits, bits = sample_bits0(foffset, *args)

    axs3[1].plot(t1, corr_d, label='dcorr_d')
    axs3[1].plot(tbits, bits, marker='x', linewidth=0)
    axs3[1].grid(True)
    #axs2[1].legend()
    axs3[1].set_xlabel('Time [sec]')

    # Extract bit transition times from here

    diff_corr_d = np.diff(corr_d)
    edges_idx = np.argwhere(np.abs(diff_corr_d) > 0.8)


    tin, tout = [0.0,], [0.0,]
    for ii, idx in enumerate(edges_idx):
        tedge = idx * fs

        # nearest bit time
        tedge2 = np.round(tedge / bitrate) * bitrate
        tin.append(tedge)
        tout.append(tedge2)
    tin.append(fs*len(pcm))
    tout.append(fs*len(pcm))

    # Function to convert actual time to desired sample time
    fsampletime = interpolate.interp1d(tin, tout, kind='quadratic')

    tbits = np.arange(nbits) * (1 / bitrate1) + tstart
    tbits = fsampletime(tbits)
    bits = fcorr(tbits)
    


    axs3[0].plot(t1, corr, label='dcorrelation')
    axs3[0].plot(tbits, bits, marker='x', linewidth=0)
    axs3[0].grid(True)





    recover_timing(pcm, t1, fs)

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


def demodulateAXCTD2(pcmin, fs):


    # TODO: we're cheating by starting after carrier received
    tstart = 7.2
    istart = int(fs * tstart)


    #basic configuration
    dt = 1/fs
    f1 = 400 # bit 1 (mark) = 400 Hz
    f2 = 800 # bit 0 (space) = 800 Hz
    bitrate = 800 #symbol rate = 800 Hz
    fprof = 7500 #7500 Hz signal associated with active profile data transmission
    pctchecksig = 0.95 #percent of each segment to check
    
    #demodulation parameter calculation
    pointsperbit = int(np.round(fs/bitrate)) #number of PCM datapoints in each bit
    window = int(np.round(pctchecksig*pointsperbit))
    tt = np.arange(window)/fs

    maxsamples = len(pcmin) // 8 #102400
    # Normalize amplitude of audio signal
    pcm = pcmin[istart:maxsamples] - np.mean(pcmin[istart:maxsamples])
    pcm *= 1.0 / np.max(pcm)
    logging.info("PCM signal length: {:d} samples, {:0.3f} seconds"
                 "".format(len(pcm), len(pcm) / fs))


    t1 = np.arange(0, len(pcm) / fs, dt) + tstart

    fig1, axs = plt.subplots(4, 1, sharex=True)


    
    # make 1200 Hz lowpass filter to separate pilot tone from digital data
    sos = signal.butter(4, 1200, btype='lowpass', fs=fs, output='sos')
    pcmlow = signal.sosfilt(sos, pcm)



    fs2 = 2400
    nsamples = fs // fs2
    #assert fs == 44100
    #q = 20
    #fs2 = fs // q # resample from 44.1 KHz to 44100

    #pcmlow2 = signal.resample(pcmlow, num=nsamples)
    #pcmhi  = pcm - pcmlow

    axs[0].plot(t1, pcmlow)
    axs[0].set_title('Digital data')
    axs[0].grid(True)


    # make highpass filter to act as a squelch.
    # if power is present, squelch.
    #sos_sq = signal.butter(2, 10000, btype='highpass', fs=fs, output='sos')
    #pcm_sq = signal.sosfilt(sos_sq, pcm)

    # starting index
    bittime = 1 / bitrate
    t2a = tstart
    t2b = t2a + bittime
    wavtime = len(pcm) / fs

    while t2b < wavtime:

        i2a = int(t2a * fs)
        i2b = int(t2b * fs)
        # waveform of this bit
        bitwfm = pcm[i2a:i2b]

        t2a = t2b
        t2b = t2a + bittime
    # end while


    # make 7000-8000 bandpass filter for pilot tone
    sos = signal.butter(6, [7400, 7600], btype='bandpass', fs=fs, output='sos')
    pcm7500hz = signal.sosfilt(sos, pcm)
    pcm7500hz /= np.max(np.abs(pcm7500hz))





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
    
    
    
    
