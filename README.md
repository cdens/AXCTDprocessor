# **AXCTDProcessor**

## Overview
AXCTDprocessor enables users to reprocess AXCTD audio (WAV) files via command line. 
Authors: Casey Densmore and Gregory Ng

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
      <td align="center">-m</td>
      <td>Profile transmission detection mode (autodetect, timefrompulse, or manual)</td>
    </tr>
    <tr>
      <td align="center">-v</td>
      <td>Enable verbose output for debugging</td>
    </tr>
  </tbody>
</table>


# Installation and Setup:
This script requires python modules other than python base. Install them with `pip install -r requirements.txt`


