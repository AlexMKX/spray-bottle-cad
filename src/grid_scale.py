"""Estimate grid pitch in px; NOT a calibrated dimensional measurement."""
import argparse
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def detect_pitch(path,band=(.53,.62)):
    im=cv2.imread(str(path))
    if im is None:raise FileNotFoundError(path)
    h,w=im.shape[:2]; y0,y1=int(h*band[0]),int(h*band[1])
    profile=np.median(cv2.cvtColor(im[y0:y1],cv2.COLOR_BGR2GRAY).astype(float),axis=0)
    highpass=profile-gaussian_filter1d(profile,19*w/1152.0)
    peaks,_=find_peaks(highpass,prominence=5.5,distance=max(2,round(15*w/1152)))
    gaps=np.diff(peaks)
    if len(gaps)<5:raise ValueError("Insufficient grid lines")
    med=np.median(gaps)
    gaps=gaps[(gaps>med*.7)&(gaps<med*1.3)]
    return {"grid_cell_px":float(np.median(gaps)),"support_intervals":len(gaps),
            "confidence":"uncalibrated, approximate"}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("photo",type=Path)
    print(detect_pitch(p.parse_args().photo))
