import numpy
import scipy.io.wavfile
import scipy.signal
from matplotlib import pyplot as plt
from scipy.fftpack import dct


def melSpectrogram(sample_rate,NFFT,nfilt):
    low_freq_mel = 0
    #将频率转换为Mel
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 生成在Mel频率的等间隔序列
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # 转换Mel频率转换为Hz频率

    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate) #获取数字频率长度

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1)))) #生成全零矩阵

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    
    #plt.plot(fbank)
    #plt.show()
    #print(fbank.shape)
    return fbank

def mfcc(audioin,fs):
    #audio_filtered=scipy.signal.filtfilt([1,0.97],1,audioin)
    AudioFiltered=numpy.append(audioin[0], audioin[1:] - 0.97 * audioin[:-1]) #信号预加重

    AudioLength=len(AudioFiltered) #信号长度
    StepLength=int(round(0.01*fs)) #步长
    FrameLength=int(round(0.025*fs)) #帧长度
    FrameN=int(numpy.ceil((AudioLength-FrameLength)/StepLength)) #计算帧的个数
    AudioLengthFixed=int(StepLength*FrameN+FrameLength) #计算可以被整除的信号长度

    zeros=numpy.zeros(AudioLengthFixed-AudioLength)
    AudioFilteredFixed=numpy.append(AudioFiltered,zeros) #将信号的长度补零，使得信号能够被整除

    indices=numpy.tile(numpy.arange(0, FrameLength), (FrameN, 1)) + numpy.tile(numpy.arange(0, FrameN * StepLength, StepLength), (FrameLength, 1)).T
    indices=numpy.mat(indices).astype(numpy.int32,copy=False)

    Frame=AudioFilteredFixed[indices]

    Frame*=numpy.hamming(FrameLength) #时域加窗

    #print(Frame.shape)

    NFFT=2048
    FRAME=abs(numpy.fft.rfft(Frame,NFFT)) #逐帧计算fft
    #print(FRAME.shape)
    #plt.plot(FRAME[100])
    #plt.show()
    FRAME=(1.0 / NFFT)*((FRAME)**2) #计算功率谱
    fbank=melSpectrogram(fs,NFFT,24) #获取梅尔滤波器组参数

    AudioMel=numpy.dot(FRAME,fbank.T) #频域滤波
    AudioMel=numpy.where(AudioMel==0,numpy.finfo(float).eps,AudioMel)#防止log10(0)的情况


    AudioMel=20*numpy.log10(AudioMel) #对数变换

    #plt.plot(AudioMel)
    #plt.show()
    #print(AudioMel.shape)
    
    mfcc=dct(AudioMel)[:,1:13] #dtc变换获取倒谱
    #plt.plot(mfcc.T)
    #plt.show()
    #print(AudioMel.shape)
    return mfcc

def main():
    sample_rate,signal=scipy.io.wavfile.read('0.wav')
    mfcc(signal,sample_rate)

if __name__ == "__main__":
    main()



