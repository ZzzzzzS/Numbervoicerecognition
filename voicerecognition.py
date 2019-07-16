import mfcc
import dtw
import scipy.io.wavfile
import wave
import time
from pyaudio import PyAudio,paInt16

class VoiceRecognition(object):
    def SystemInit(self):

        #输入模板信号
        self.sample_rate,self.signal1=scipy.io.wavfile.read('1.wav')
        self.sample_rate,self.signal2=scipy.io.wavfile.read('2.wav')
        self.sample_rate,self.signal3=scipy.io.wavfile.read('3.wav')
        self.sample_rate,self.signal4=scipy.io.wavfile.read('4.wav')
        self.sample_rate,self.signal5=scipy.io.wavfile.read('5.wav')
        self.sample_rate,self.signal6=scipy.io.wavfile.read('6.wav')
        self.sample_rate,self.signal7=scipy.io.wavfile.read('7.wav')
        self.sample_rate,self.signal8=scipy.io.wavfile.read('8.wav')
        self.sample_rate,self.signal9=scipy.io.wavfile.read('9.wav')
        self.sample_rate,self.signal0=scipy.io.wavfile.read('0.wav')

        self.sample_rate,self.signalAuther=scipy.io.wavfile.read('Auther.wav')

        #对模板信号求mfcc
        self.mfcc0=mfcc.mfcc(self.signal0,self.sample_rate)
        self.mfcc1=mfcc.mfcc(self.signal1,self.sample_rate)
        self.mfcc2=mfcc.mfcc(self.signal2,self.sample_rate)
        self.mfcc3=mfcc.mfcc(self.signal3,self.sample_rate)
        self.mfcc4=mfcc.mfcc(self.signal4,self.sample_rate)
        self.mfcc5=mfcc.mfcc(self.signal5,self.sample_rate)
        self.mfcc6=mfcc.mfcc(self.signal6,self.sample_rate)
        self.mfcc7=mfcc.mfcc(self.signal7,self.sample_rate)
        self.mfcc8=mfcc.mfcc(self.signal8,self.sample_rate)
        self.mfcc9=mfcc.mfcc(self.signal9,self.sample_rate)

        self.mfccAuther=mfcc.mfcc(self.signalAuther,self.sample_rate)

    def GetNumber(self,AudioIn,Fs):

        #对输入信号求mfcc
        ReceivedSignal=mfcc.mfcc(AudioIn,Fs)

        value=[] #开始进行模板匹配
        value.append(dtw.dtw(self.mfcc0,ReceivedSignal))
        value.append(dtw.dtw(self.mfcc1,ReceivedSignal))
        value.append(dtw.dtw(self.mfcc2,ReceivedSignal))
        value.append(dtw.dtw(self.mfcc3,ReceivedSignal))
        value.append(dtw.dtw(self.mfcc4,ReceivedSignal))
        value.append(dtw.dtw(self.mfcc5,ReceivedSignal))
        value.append(dtw.dtw(self.mfcc6,ReceivedSignal))
        value.append(dtw.dtw(self.mfcc7,ReceivedSignal))
        value.append(dtw.dtw(self.mfcc8,ReceivedSignal))
        value.append(dtw.dtw(self.mfcc9,ReceivedSignal))

        value.append(dtw.dtw(self.mfccAuther,ReceivedSignal))

        #print(value)
        number=value.index(min(value)) #获取最接近的值
        #print(number)
        return number

    def VoiceRecord(self):

        print("开始录音")
          
        framerate=44100
        NUM_SAMPLES=441
        channels=1
        sampwidth=2

        pa=PyAudio()
        stream=pa.open(format = paInt16,channels=1,rate=framerate,input=True,frames_per_buffer=NUM_SAMPLES)
        my_buf=[]
        count=0
        while count<200:#控制录音时间
            string_audio_data = stream.read(NUM_SAMPLES)
            my_buf.append(string_audio_data)
            count+=1
            #print(count)

        wf=wave.open("SignalIn.wav",'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(b"".join(my_buf))
        wf.close()
        stream.close()
        self.sample_rate,self.signalinput=scipy.io.wavfile.read('SignalIn.wav')

        print("录音结束")


def VoiceRecognitionNoGUI():
    print("系统初始化中")
    a=VoiceRecognition()
    a.SystemInit()
    print("系统初始化完成")
    time.sleep(1)
    a.VoiceRecord()
    print("开始分析")
    number=a.GetNumber(a.signalinput,a.sample_rate)
    print("分析完成")
    if number==10:
        print("Powered by ZZS 2019©All Rights Reserved")
        print("本程序由160200531周子顺制作，欢迎使用")
    else:
        print(number)
        

def main():
    VoiceRecognitionNoGUI()


if __name__ == "__main__":
    main()