from mfcc import mfcc
import numpy
import scipy.io.wavfile


def dtw(t,r):
    n=len(t) #获取信号长度
    m=len(r)
    d=numpy.zeros((n,m)) #生成零矩阵
    for i in range(n):
        for j in range(m):
            d[i,j]=numpy.linalg.norm(t[i,:]-r[j,:])
    D=numpy.zeros((n,m))

    D[0,0]=d[0,0]
    D[0,1]=d[0,1]
    D[1,0]=d[1,0]
    for i in range(n-1):
        for j in range(m-1):
            D[i+1,j+1]=min((D[i+1,j]+d[i+1,j+1]),(D[i,j+1]+d[i+1,j+1]),(D[i,j]+d[i+1,j+1])) 
            #print(D[i+1,j+1])
        #print('aaas')
    #numpy.set_printoptions(threshold = 1e6)
    #print(D)
    return D[-1,-1]



def main():
    sample_rate,signal11=scipy.io.wavfile.read('1.wav')
    sample_rate,signal12=scipy.io.wavfile.read('2.wav')
    sample_rate,signal13=scipy.io.wavfile.read('3.wav')
    sample_rate,signal14=scipy.io.wavfile.read('4.wav')
    sample_rate,signal15=scipy.io.wavfile.read('5.wav')
    sample_rate,signal16=scipy.io.wavfile.read('6.wav')
    sample_rate,signal17=scipy.io.wavfile.read('7.wav')
    sample_rate,signal18=scipy.io.wavfile.read('8.wav')
    sample_rate,signal19=scipy.io.wavfile.read('9.wav')
    sample_rate,signal10=scipy.io.wavfile.read('0.wav')

    print('ok1')

    mfcc10=mfcc(signal10,sample_rate)
    mfcc11=mfcc(signal11,sample_rate)
    mfcc12=mfcc(signal12,sample_rate)
    mfcc13=mfcc(signal13,sample_rate)
    mfcc14=mfcc(signal14,sample_rate)
    mfcc15=mfcc(signal15,sample_rate)
    mfcc16=mfcc(signal16,sample_rate)
    mfcc17=mfcc(signal17,sample_rate)
    mfcc18=mfcc(signal18,sample_rate)
    mfcc19=mfcc(signal19,sample_rate)

    print('ok2')
    value=[]
    
    sample_rate,signalinput=scipy.io.wavfile.read('1.wav')
    ReceivedSignal=mfcc(signalinput,sample_rate)

    value.append(dtw(mfcc10,ReceivedSignal))
    value.append(dtw(mfcc11,ReceivedSignal))
    value.append(dtw(mfcc12,ReceivedSignal))
    value.append(dtw(mfcc13,ReceivedSignal))
    value.append(dtw(mfcc14,ReceivedSignal))
    value.append(dtw(mfcc15,ReceivedSignal))
    value.append(dtw(mfcc16,ReceivedSignal))
    value.append(dtw(mfcc17,ReceivedSignal))
    value.append(dtw(mfcc18,ReceivedSignal))
    value.append(dtw(mfcc19,ReceivedSignal))

    

    print(value)
    number=value.index(min(value))
    print(number)

if __name__ == "__main__":
    main()