# **AXCTD Processor**

## Overview
AXCTDprocessor enables users to reprocess AXCTD audio (WAV) files via command line. 

Authors: Casey Densmore and Gregory Ng

The code is written in a manner (with buffers and a while loop) to make conversion to a version suitable for realtime processing with a VHF receiver simple.


<br />
<br />
## Usage:
`python3 processAXCTD -i inputfile.WAV`

### Optional flags:
<table>
  <tbody>
    <tr>
      <th align="center">Flag</th>
      <th align="left">Purpose</th>
    </tr>
    <tr>
      <td align="center">-o</td>
      <td>Output Filename Header: directory and header for output debug and profile files</td>
    </tr>
    <tr>
      <td align="center">-s</td>
      <td>Start time of AXCTD profile in WAV file, format: SS, MM:SS, or HH:MM:SS (default: 0 sec)</td>
    </tr>
    <tr>
      <td align="center">-e</td>
      <td>End time of AXCTD profile in WAV file, format: SS, MM:SS, or HH:MM:SS (default: end of WAV file)</td>
    </tr>
    <tr>
      <td align="center">-a</td>
      <td>Minimum number of seconds after first 400 Hz pulse is detected that profile initiation can be triggered</td>
    </tr>
    <tr>
      <td align="center">-b</td>
      <td>Maximum number of seconds after first 400 Hz pulse is detected that profile initiation is triggered (if no 7500 Hz tone is detected, the profile will be triggered at this time)</td>
    </tr>
    <tr>
      <td align="center">-p</td>
      <td>The threshold of the 400 Hz signal level normalized to the dead frequency (selectable) signal level to recognize a 400 Hz "header" pulse</td>
    </tr>
    <tr>
      <td align="center">-t</td>
      <td>The threshold of the 7500 Hz signal level normalized to the dead frequency (selectable) signal level to recognize that profile transmission has begun</td>
    </tr>
    <tr>
      <td align="center">-d</td>
      <td>The dead frequency- a quiet frequency used to normalize probe transmission signal levels to background static noise</td>
    </tr>
    <tr>
      <td align="center">-l</td>
      <td>The number of audio PCM datapoints processed per loop (minimum of 1-2 times the sampling frequency of the data is recommended)</td>
    </tr>
  </tbody>
</table>

<br />
<br />

# Installation and Setup:
This script requires python modules other than python base. Install them with `pip install -r requirements.txt`


<br />
<br />
## AXCTD Overview

Airborne eXpendable Conductivity Temperature Depth (AXCTD) probes are air-launched, single use probes which transmit temperature and conductivity data via a VHF FM radio frequency. Depth is estimated using an assumed sink rate of the probe through the water column, and salinity (and optionally density and sound speed) is calculated as a function of the temperature, conductivity, and depth.

AXCTDs transmit their data via a frequency shift key (FSK) modulation scheme with a symbol rate of 800 Hz, using a mark ('1' bit) frequency of 400 Hz and a space ('0' bit) frequency of 800 Hz. Data are transmitted as a series of 32-bit frames, each of which contains one temperature and salinity datapoint as well as a cyclic redundancy check (described below). Given the bit rate of 800 Hz and frame length of 32 bits, this means the AXCTD transmits 25 datapoints (frames) per second.

An AXCTD transmission begins with approximately 2 seconds of a pure 400 Hz tone, followed by 72 frames (2.8 seconds) of header data. This data includes probe type ('A000' for AXCTD), serial number, and maximum depth (m), as well as temperature, conductivity, and depth conversion coefficients which may vary between probes. This header is followed by a 5 second gap, and the process repeats twice. After the third header, the AXCTD begins transmitting temperature and conductivity data as the probe sits at the surface. When the probe is released, the AXCTD adds a 7500 Hz tone, which continues for the duration of profile collection.

Each AXCTD data frame is 32 bits and follows a standard format. For example, <b>*10011100100001001000011111011110*</b> is broken down as: 

|10 |   011100100001 | 001000011111 | 011110 |
| :---: |:---:|:---: | :---:| 
| Header | Conductivity | Temperature	|  CRC |

The frame components are:

- Header- the first two bits of a frame are always <b>10</b>
- Conductivity- the next 12 bits (positions 3-14) contain the conductivity datapoint
- Temperature- the next 12 bits (positions 15-26) contain the temperature datapoint
- Cyclic redundancy check- the last 6 bits (positions 27-32) contain a cyclic reduncancy check with a 7-bit polynomial divisor (decimal representation 101)

The temperature and conductivity integers conveyed in each frame are translated first into uncalibrated values in degrees Celsius and mS/cm (respectively)- temperatures via a lookup table and conductivity via a simple equation: <br />
C<sub>uncal</sub> = C<sub>integer</sub> * 60/4096

Depth and calibrated temperature and conductivity are calculated from probe descent time and uncalibrated temperature and conductivity (respectively) via a cubic equation using the coefficients transmitted in the AXCTD's header message. For example, given the four depth coefficients D<sub>0</sub> through D<sub>3</sub>, depth is calculated from time as: <br />
z = D<sub>0</sub> + D<sub>1</sub> * t + D<sub>2</sub> * t<sup>2</sup> + D<sub>3</sub> * t<sup>3</sup>



<br />
<br />

## AXCTD Processor Implementation

Processing AXCTDs comes with several challenges:

 - Identifying the 400 Hz pulses and 7500 Hz tones
 - Demodulating the FSK modulated bitstream
 - Decoding the header data
 - Decoding the profile

To identify the 400 Hz and 7500 Hz tones, the AXCTDprocessor calculates the signal levels at each frequency at a rate of 25 Hz (once per frame). These signal levels are normalized by the signal level at an expected quiet frequency (default 3000 Hz) and smoothed. Minimum signal ratios necessary to detect 400 Hz and 7500 Hz tones can be specified by the user. 

The FSK demodulation implemented in the AXCTDProcessor is (relatively) simple: first, the PCM data are smoothed via a 1200 Hz low pass filter. Next, all zero-crossing points are identified, and filtered to identify zero-crossing points that match the 800 Hz symbol rate. For each expected bit, the 400 Hz and 800 Hz powers are calculated via a Fourier transform over the length of the bit, and the greater of the two powers is used to classify each bit as mark or space (1 or 0). 

To account for potential differences in amplitude of the 400 Hz and 800 Hz signals due to probe variations or other effects, ratios of 400 Hz and 800 Hz power levels are calculated for each bit in the series of header data following the first pulse. A cumulative probability distribution function is used to split the power ratios into two nearly equal groups (approximately 40% 400 Hz and 60% 800 Hz signals, based on expected distributions) and the scaling necessary to acoomplish that is used for all subsequent demodulation. The second and third headers are demodulated and parsed to decode AXCTD metadata such as serial numbers and conversion coefficients, and data from the last two headers are compared so any information missed due to demodulation errors in the second header can be incorporated from the third and vice versa.

Both header and profile data parsing are accomplished by identifying 32-bit sequences beginning with <b>10</b> that have a valid CRC, and converting the data from there. Although some random variations can (and do) pass the CRC, these are rare enough that it does not affect processing success. Error code corrections are not applied in this algorithm; any frames with bit errors are simply ignored.

After each profile frame is parsed, temperature and conductivity are calculated from the frame and the corresponding depth is calculated via the time difference between the start of the frame and the first detection of the 7500 Hz signal. Salinity is then calculated using the temperature, conductivity, and depth for that frame.

The AXCTDProcessor is written in a seemingly unnecessarily complicated manner, with buffers for PCM data and the demodulated bitstream as the program processes a preset number of PCM points via each iteration (set via the pointsperloop flag <b>-l</b>, default 100000). Although a simpler version can be (and was) written that demodulates the entire profile first, and then parses the bitstream into valid header and profile data, the AXCTDProcessor was developed in this manner to simplify translation into a realtime processor connected to a VHF radio receiver that provides immediate feedback to a user processing profiles in an aircraft.