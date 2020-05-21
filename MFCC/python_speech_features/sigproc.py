
import decimal


import numpy
import math


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def framesig(sig,frame_len,frame_step,winfunc=lambda x:numpy.ones((x,))):
   
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0*slen - frame_len)/frame_step))

    padlen = int((numframes-1)*frame_step + frame_len)

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig,zeros))

    indices = numpy.tile(numpy.arange(0,frame_len),(numframes,1)) + numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = numpy.array(indices,dtype=numpy.int32)
    frames = padsignal[indices]
    win = numpy.tile(winfunc(frame_len),(numframes,1))

    #print(numframes)
    return frames*win


def deframesig(frames,siglen,frame_len,frame_step,winfunc=lambda x:numpy.ones((x,))):
    
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'
 
    indices = numpy.tile(numpy.arange(0,frame_len),(numframes,1)) + numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = numpy.array(indices,dtype=numpy.int32)
    padlen = (numframes-1)*frame_step + frame_len   
    
    if siglen <= 0: siglen = padlen
    
    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    win = winfunc(frame_len)
    
    for i in range(0,numframes):
        window_correction[indices[i,:]] = window_correction[indices[i,:]] + win + 1e-15 
        rec_signal[indices[i,:]] = rec_signal[indices[i,:]] + frames[i,:]
        
    rec_signal = rec_signal/window_correction
    return rec_signal[0:siglen]
    
def magspec(frames,NFFT):
    
    complex_spec = numpy.fft.rfft(frames,NFFT)
    return numpy.absolute(complex_spec)
          
def powspec(frames,NFFT):
    
    return 1.0/NFFT * numpy.square(magspec(frames,NFFT))
    
def logpowspec(frames,NFFT,norm=1):
     
    ps = powspec(frames,NFFT);
    ps[ps<=1e-30] = 1e-30
    lps = 10*numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps
    
def preemphasis(signal,coeff=0.95):
    
    return numpy.append(signal[0],signal[1:]-coeff*signal[:-1])



