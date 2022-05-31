#! /usr/bin/env/python3
# Casey R. Densmore 15JAN2022
# Modified version of ARES to process AXBT audio files programatically


import numpy as np
from scipy.io import wavfile #for wav file reading
from scipy import signal

import wave #WAV file writing

import time as timemodule
import datetime as dt

from traceback import print_exc as trace_error

from shutil import copy as shcopy
from sys import getsizeof

import demodulate, parse


#this could be problematic for debugging but I'm getting tired of the "chunk could not be understood" messages
import warnings
warnings.filterwarnings("ignore")


#reading the audio file
def readAXCTDwavfile(inputfile):
    
    #reading WAV file
    fs, snd = wavfile.read(inputfile)
    
    #if multiple channels, sum them together
    sndshape = np.shape(snd) #array size (tuple)
    ndims = len(sndshape) #number of dimensions
    if ndims == 1: #if one channel, use that
        audiostream = snd
    elif ndims == 2: #if two channels
        #audiostream = np.sum(snd,axis=1) #sum them up
        audiostream = snd[:,0] #use first channel
    else:
        raise Exception('Too many dimensions for an audio file!')
    
    # Normalize amplitude/DC offset of audio signal
    pcm_dc = np.mean(audiostream)
    pcm_ampl = np.max(np.abs(audiostream))
    pcm = (audiostream.astype(np.float) - pcm_dc) / pcm_ampl
        
    # downsampling if necessary 
    if fs > 50000: 
        pcm = signal.decimate(pcm, 2)
        fs /= 2
        
    return pcm, fs
    


    
    
# =============================================================================
#  AXCTD Processor class
# =============================================================================


class AXCTD_Processor:

    #initializing current thread (saving variables, reading audio data or contacting/configuring receiver)
    def __init__(self, audiofile, minR400=2.0, mindR7500=1.5, deadfreq=3000, pointsperloop=88200, timerange=[0,-1], triggerrange=[30,-1]):
        
        #prevents Run() method from starting before init is finished (value must be changed to 100 at end of __init__)
        self.startthread = 0 
        
        self.keepgoing = True  # signal connections
        self.waittoterminate = False #whether to pause on termination of run loop for kill process to complete
        
        self.audiofile = audiofile
        
        self.minR400 = minR400 #threshold to ID first 400 Hz pulse
        self.mindR7500 = mindR7500 #threshold to ID profile start by 7.5 kHz tone
        self.deadfreq = deadfreq #frequency to use as "data-less" control to normalize signal levels
        self.pointsperloop = pointsperloop #how many PCM datapoints AXCTDprocessor handles per loop
        
        #index 0: earliest time AXBT will trigger after 400 Hz pulse in seconds (default 30 sec)
        #index 1: time AXCTD will autotrigger without 7.5kHz signal (set to -1 to never trigger profile)
        self.triggerrange = triggerrange
        
        #lists to save output data
        self.temperature = []
        self.conductivity = []
        self.salinity = []
        self.depth = []
        self.time = []
        self.hex_frame = []
        
        self.past_headers = False #when false, program may try to read header data
        self.header1_read = False #notes when each header has been successfully read, to stop trying
        self.header2_read = False
        self.header3_read = False
        
        #temperature lookup table, calibration coefficients
        self.metadata = parse.initialize_axctd_metadata()
        self.metadata['counter_found_2'] = [False] * 72
        self.metadata['counter_found_3'] = [False] * 72
        self.tempLUT = parse.read_temp_LUT('temp_LUT.txt')
        self.tcoeff = self.metadata['tcoeff_default']
        self.ccoeff = self.metadata['ccoeff_default']
        self.zcoeff = self.metadata['zcoeff_default']
        
        #store powers at different frequencies used to ID profile start
        self.p400 = np.array([])
        self.p7500 = np.array([])
        self.pdead = np.array([])
        self.r400 = np.array([])
        self.r7500 = np.array([])
        self.power_inds = []
        
        self.firstpulse400 = -1 #will store r400 index corresponding to first 400 Hz pulse
        self.curpulse = 0
        self.mean7500pwr = np.NaN
        self.profstartind = -1
        self.lastdemodind = -1 #set to -1 to indicate no demodulation has occurred
        
        #reading in PCM data
        self.audiostream, self.fs = readAXCTDwavfile(self.audiofile)
        
        #trimming PCM data to specified range as required
        if timerange[1] > 0: #trim end of profile first to avoid throwing off indicies
            e = int(self.fs*timerange[1])
            self.audiostream = self.audiostream[:e]
        if timerange[0] > 0:
            s = int(self.fs*timerange[0])
            self.audiostream = self.audiostream[s:]
                    
        self.numpoints = len(self.audiostream)
        
        #constant settings for analysis
        self.fs_power = 25 #check 25 times per second aka once per frame
        self.N_power = int(self.fs/10) #each power calculation is for 0.1 seconds of PCM data
        self.power_smooth_window = 5
        self.d_pcm = int(np.round(self.fs/self.fs_power)) #how many points apart to sample power
        
        self.demod_buffer = np.array([]) #buffer (list) for demodulating PCM data
        self.demodbufferstartind = 0
        self.demod_Npad = 50 #how many points to pad on either side of demodulation window (must be larger than window length for low-pass filter in demodulation function)
        
        self.high_bit_scale = 1.5 #scale factor for high frequency bit to correct for higher power at low frequencies (will be adjusted to optimize demodulation after reading first header data)
        
        self.binary_buffer = [] #buffer for demodulated binary data not organized into frames
        self.binary_buffer_inds = [] #pcm indices for each bit start point (used to calculate observation times during frame parsing)
        self.binary_buffer_conf = [] #confidence ratios: used for demodulation debugging/improvement
        self.r7500_buffer = [] #holds 7500 Hz sig lev ratios corresponding to each bit
        
        #contains all binary data
        self.bindata = []
        self.bininds = [] #PCM indices for the start of each bit
        self.binconf = [] #confidence in bit ID
        
        #trig terms for power calculations
        self.theta400 = 2*np.pi*np.arange(0,self.N_power)/self.fs*400 #400 Hz (main pulse)
        self.theta7500 = 2*np.pi*np.arange(0,self.N_power)/self.fs*7500 #400 Hz (main pulse)
        self.thetadead = 2*np.pi*np.arange(0,self.N_power)/self.fs*self.deadfreq #400 Hz (main pulse)
        
        #-1: not processing, 0: no pulses, 1: found pulse, 2: active profile parsing
        self.status = -1 
                
        
    def run(self):
        
        
        # setting up thread while loop- terminates when user clicks "STOP" or audio file finishes processing
        self.status = 0
        
        #initializing audio buffer: self.pcmind = index of first point, demod_buffer contains numpy array of pcm data waiting to be demodulated
        self.bufferstartind = 0
        
        
        #MAIN PROCESSOR LOOP
        while self.keepgoing:    
            
            #calculating end of next slice of PCM data for signal level calcuation and demodulation
            e = self.demodbufferstartind + self.pointsperloop
            
            if self.numpoints - self.demodbufferstartind < self.fs: #within 1 second of file end
                self.keepgoing = False
                return
            elif e >= self.numpoints: #terminate loop if at end of file
                e = self.numpoints - 1
            
            #add next round of PCM data to buffer for signal calculation and demodulation
            # self.demod_buffer = np.append(self.demod_buffer, self.audiostream[self.demodbufferstartind:e])
            self.demod_buffer = self.audiostream[self.demodbufferstartind:e]
            
            #getting signal levels at 400 Hz, 7500 Hz, and dead frequency
            #we don't have to calculate all three every time but need to calculate at least two of them every
            #time and the logic/code complexity to save computing an unnecessary one isn't worth the computing
            #power saved
            #sampling interval = sampling frequency / power sampling frequency
            
            pstartind = len(self.power_inds)
                
            #calculating signal levels at 400 Hz, 7500 Hz, and dead frequency (default 3000 Hz)
            self.power_inds.extend([ind for ind in range(self.demodbufferstartind, e-self.N_power, self.d_pcm)])
            for cind in self.power_inds[pstartind:]:
                bufferind = cind - self.demodbufferstartind
                cdata = self.demod_buffer[bufferind:bufferind+self.N_power]
                self.p400 = np.append(self.p400, np.abs(np.sum(cdata*np.cos(self.theta400) + 1j*cdata*np.sin(self.theta400))))
                self.p7500 = np.append(self.p7500, np.abs(np.sum(cdata*np.cos(self.theta7500) + 1j*cdata*np.sin(self.theta7500))))
                self.pdead = np.append(self.pdead, np.abs(np.sum(cdata*np.cos(self.thetadead) + 1j*cdata*np.sin(self.thetadead))))
                        
            #smoothing signal levels, calculating R400/R7500
            self.p400 = demodulate.boxsmooth_lag(self.p400, self.power_smooth_window, pstartind)
            self.p7500 = demodulate.boxsmooth_lag(self.p7500, self.power_smooth_window, pstartind)
            self.pdead = demodulate.boxsmooth_lag(self.pdead, self.power_smooth_window, pstartind)
            self.r400 = np.append(self.r400, np.log10(self.p400[pstartind:]/self.pdead[pstartind:]))
            self.r7500 = np.append(self.r7500, np.log10(self.p7500[pstartind:]/self.pdead[pstartind:]))
            
            
            #look for 400 Hz pulse if it hasn't been discovered yet
            if self.status == 0:
                matchpoints = np.where(self.r400[pstartind:] >= self.minR400)
                if len(matchpoints[0]) > 0: #found a match!
                    self.firstpulse400 = self.power_inds[pstartind:][matchpoints[0][0]] #getting index in original PCM data
                    self.status = 1
                
            
            #if pulse discovered, demodulate data to bitstream AND check 7500 Hz power
            if self.status >= 1:
                
                #calculate 7500 Hz lower power (in range 4.5-5.5 seconds after first 400 Hz pulse start)
                #last power index is at least 5.5 seconds after 400 Hz pulse detected
                if self.power_inds[-1] >= self.firstpulse400 + int(self.fs*5.5) and np.isnan(self.mean7500pwr): 
                    #mean signal level at 7500 Hz between 4.5 and 5.5 sec after 400 Hz pulse
                    pwr_ind_array = np.asarray(self.power_inds)
                    s7500ind = np.argmin(np.abs(self.firstpulse400 + int(self.fs*4.5) - pwr_ind_array))
                    e7500ind = np.argmin(np.abs(self.firstpulse400 + int(self.fs*5.5) - pwr_ind_array))
                    self.mean7500pwr = np.nanmean(self.r7500[s7500ind:e7500ind])
                
                #get 7500 Hz power and update status (and bitstream) as necessary 
                #(only bother if time since 400 Hz pulse exceeds the minimum time for profile start)
                if self.power_inds[-1] > self.firstpulse400 + int(self.triggerrange[0]*self.fs):
                    if not np.isnan(self.mean7500pwr) and self.status == 1:
                        matchpoints = np.where(self.r7500[pstartind:] - self.mean7500pwr >= self.mindR7500)
                        if len(matchpoints[0]) > 0: #if power threshold is exceeded
                            self.profstartind = self.power_inds[pstartind:][matchpoints[0][0]]
                            self.status = 2
                    #if the latest trigger time setting has been exceeded
                    elif self.triggerrange[1] > 0 and self.power_inds[-1] >= self.firstpulse400 + int(self.fs*self.triggerrange[1]):
                        self.profstartind = self.firstpulse400 + int(self.fs*self.triggerrange[1])
                        self.status = 2
                
                #demodulate to bitstream and append bits to buffer
                curbits, conf, bit_edges, next_demod_ind = demodulate.demodulate_axctd(self.demod_buffer, self.fs, self.demod_Npad, self.high_bit_scale)
                
                self.binary_buffer.extend(curbits) #buffer for demodulated binary data not organized into frames
                
                new_bit_inds = [be + self.demodbufferstartind for be in bit_edges]
                self.binary_buffer_inds.extend(new_bit_inds)
                
                self.binary_buffer_conf.extend(conf)
                
                #saving for file writing
                self.bindata.extend(curbits)
                self.bininds.extend(new_bit_inds)
                self.binconf.extend(conf)
                
                #array of profile signal levels to go with other data 
                recent_r7500 = self.r7500[pstartind:]
                recent_pwrinds = self.power_inds[pstartind:]
                new_r7500 = [recent_r7500[np.argmin(np.abs(recent_pwrinds - ci))]-self.mean7500pwr for ci in new_bit_inds]
                self.r7500_buffer.extend(new_r7500)
                
                
                #attempting to read headers for conversion coefficients and AXCTD metadata
                if self.status >= 1 and not self.past_headers:
                
                    #seeing if enough time has passed since first pulse to contain 2nd or 3rd header data
                    #pulse length: 1.8 sec, header length: 2.88 sec, gap period (first 2 pulses): 5 sec
                    #total pulse cycle ~= 9.68 sec (assume 9-10 sec)
                    
                    headerdata = [None,None]
                    
                    firstbin = self.binary_buffer_inds[0]
                    lastbin = self.binary_buffer_inds[-1]
                    cbufferindarray = np.asarray(self.binary_buffer_inds)
                    
                    #first header should start around 1.8 sec and end around 3.7 seconds
                    #only processing a small margin within that to be sure we are only capturing 1 sec of header
                    p1headerstartpcm = self.firstpulse400 + int(self.fs*2.3)
                    p1headerendpcm = self.firstpulse400 + int(self.fs*3.3)
                    
                    #second header should start around 11.48 sec and end around 14.36 seconds
                    p2headerstartpcm = self.firstpulse400 + int(self.fs*10.5) #capture 1 sec of pulse
                    p2headerendpcm = self.firstpulse400 + int(self.fs*14.8) #~half second margin on backend
                    
                    #third header should start around 21.16 sec and end around 24.04 seconds
                    p3headerstartpcm = self.firstpulse400 + int(self.fs*20) #same margins as header 2
                    p3headerendpcm =  self.firstpulse400 + int(self.fs*24.5)
                    
                    #establishing signal amplitudes based on first header to improve demodulation
                    if firstbin <= p1headerstartpcm and lastbin >= p1headerendpcm and not self.header1_read: 
                        
                        #determining binary data start/end index (adding extra 0.5 sec of data if available)
                        p1startind = np.where(cbufferindarray >= p1headerstartpcm - int(self.fs*0.5))[0][0]
                        p1endind = np.where(cbufferindarray <= p1headerendpcm + int(self.fs*0.5))[0][-1]
                        
                        #pulling confidence ratios from the header and recalculating optimal high bit scale
                        header_confs = self.binary_buffer_conf[p1startind:p1endind]
                        self.high_bit_scale = demodulate.adjust_scale_factor(header_confs, self.high_bit_scale)
                        self.header1_read = True
                        
                    
                    #trying to process second header
                    if firstbin <= p2headerstartpcm and lastbin >= p2headerendpcm and not self.header2_read: 
                        
                        #determining binary data start/end index (adding extra 0.5 sec of data if available)
                        p2startind = np.where(cbufferindarray >= p2headerstartpcm - int(self.fs*0.5))[0][0]
                        p2endind = np.where(cbufferindarray <= p2headerendpcm + int(self.fs*0.5))[0][-1]
                        
                        #pulling header data from pulse
                        header_bindata = parse.trim_header(self.binary_buffer[p2startind:p2endind])
                        
                        if len(header_bindata) >= 72*32: #must contain full header
                        
                            #parsing and converting header information
                            headerdata[0] = parse.parse_header(header_bindata)
                            self.header2_read = True #read complete for 2nd header transmission
                            
                    #trying to process third header
                    if firstbin <= p3headerstartpcm and lastbin >= p3headerendpcm and not self.header3_read: 
                        
                        #determining binary data start/end index (adding extra 0.5 sec of data if available)
                        p3startind = np.where(cbufferindarray >= p3headerstartpcm - int(self.fs*0.5))[0][0]
                        p3endind = np.where(cbufferindarray <= p3headerendpcm + int(self.fs*0.5))[0][-1]
                        
                        #pulling header data from pulse
                        header_bindata = parse.trim_header(self.binary_buffer[p3startind:p3endind])
                        
                        if len(header_bindata) >= 72*32: #must contain full header
                        
                            #parsing and converting header information
                            headerdata[1] = parse.parse_header(header_bindata)
                            self.header3_read = True #read complete for 3rd header transmission
                            
                            
                        
                    #incorporating AXCTD header info into profile metadata
                    coeffs = ['t','c','z']
                    other_data = ['serial_no','probe_code','max_depth','misc']
                    for i,header in enumerate(headerdata):
                        
                        if header is not None:
                            
                            self.metadata[f'frame_data_{i+2}'] = header['frame_data']
                            self.metadata[f'counter_found_{i+2}'] = header['counter_found']
                            
                            for coeff in coeffs: #incorporating coefficients (coeff, coeff_valid, coeff_hex)
                                for ci in range(4):
                                    if header[coeff + 'coeff_valid'][ci]:
                                        self.metadata[coeff + 'coeff'][ci] = header[coeff + 'coeff'][ci]
                                        self.metadata[coeff + 'coeff_hex'][ci] = header[coeff + 'coeff_hex'][ci]
                                        self.metadata[coeff + 'coeff_valid'][ci] = True
                            
                            for key in other_data: #incorporating other profile metadata
                                if header[key] is not None and self.metadata[key] is None:
                                    self.metadata[key] = header[key]
                                    
                    
                    #if updated headers included, then try to update coefficients
                    if headerdata[0] is not None or headerdata[1] is not None: 
                        if sum(self.metadata['tcoeff_valid']) == 4:
                            self.tcoeff = self.metadata['tcoeff']
                        if sum(self.metadata['ccoeff_valid']) == 4:
                            self.ccoeff = self.metadata['ccoeff']
                        if sum(self.metadata['tcoeff_valid']) == 4:
                            self.zcoeff = self.metadata['zcoeff']
                    
                    
                if self.status == 2: #parsing bitstream into frames and calculating updated profile data
                    
                    self.past_headers = True
                    
                    #cutting off all data before profile initiation
                    if self.binary_buffer_inds[0] <= self.profstartind:
                        firstind = np.where(np.asarray(self.binary_buffer_inds) > self.profstartind)[0][0]
                        self.binary_buffer = self.binary_buffer[firstind:]
                        self.binary_buffer_inds = self.binary_buffer_inds[firstind:]
                        self.binary_buffer_conf = self.binary_buffer_conf[firstind:]
                        self.r7500_buffer = self.r7500_buffer[firstind:]
                    
                    #calculting times corresponding to each bit
                    binbufftimes = (np.asarray(self.binary_buffer_inds) - self.profstartind)/self.fs
                        
                    #parsing data into frames
                    hexframes,times,depths,temps,conds,psals,next_buffer_ind = parse.parse_bitstream_to_profile(self.binary_buffer, binbufftimes, self.r7500_buffer, self.tempLUT, self.tcoeff, self.ccoeff, self.zcoeff)
                    
                    #rounding data and appending to lists
                    self.hex_frame.extend(hexframes)
                    self.time.extend(np.round(times,2))
                    self.depth.extend(np.round(depths,2))
                    self.temperature.extend(np.round(temps,2))
                    self.conductivity.extend(np.round(conds,2))
                    self.salinity.extend(np.round(psals,2))
                    
                    #removing parsed data from binary buffer
                    self.binary_buffer = self.binary_buffer[next_buffer_ind:]
                    self.binary_buffer_inds = self.binary_buffer_inds[next_buffer_ind:]
                    self.r7500_buffer = self.r7500_buffer[next_buffer_ind:]
                                    
            #increment demod buffer index forward, kill process once time exceeds the max time of the audio file
            if self.status > 0: #if active demodulation occuring, increment to start of next bit, corrected for padding
                if next_demod_ind > self.demod_Npad:
                    self.demodbufferstartind += next_demod_ind - self.demod_Npad
                else:
                    print("[!] ERROR: next_demod_ind = demod_Npad, unable to continue (try increasing pointsperloop to 1-2 times sampling frequency!")
                    exit()
                    
            else: #signal level calcs only, increment buffer by pointsperloop
                self.demodbufferstartind += self.pointsperloop                
            
            print(f"[+] Processing status: {round(100*self.demodbufferstartind/self.numpoints)}%         ", end='\r')


        
        
        