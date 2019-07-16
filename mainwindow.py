import time
import wave
from pyaudio import PyAudio,paInt16
import sys
from PySide2.QtWidgets import QApplication, QMainWindow
from UI_mainwindow import Ui_MainWindow
import voicerecognition
import scipy.io.wavfile

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.VoiceHandle=voicerecognition.VoiceRecognition()
        self.ui.pushButton.setEnabled(False)
        self.ui.pushButton.setText("系统初始化中")
        self.ui.pushButton.clicked.connect(self.OnpushButtonClicked)
        self.ui.textEdit.setReadOnly(True)
        self.setFixedSize(300,600)
        self.setWindowTitle('数字识别系统')

    
    def InitVoiceHandle(self):
        self.VoiceHandle.SystemInit()
        self.ui.pushButton.setEnabled(True)
        self.ui.pushButton.setText("开始")

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
    
    def VoiceAnalyse(self):

        print("正在分析")

        number=self.VoiceHandle.GetNumber(self.signalinput,self.sample_rate)

        if number==10:
            self.ui.textEdit.append("Powered by ZZS")
            self.ui.textEdit.append("2019©All Rights Reserved")
            self.ui.textEdit.append("本程序由160200531周子顺制作，欢迎使用")
        else:
            self.ui.textEdit.append(str(number))
        
        print("分析完成")

    
    def OnpushButtonClicked(self):
        
        self.ui.pushButton.setEnabled(False)
        self.ui.pushButton.setText("正在录音")

        QApplication.processEvents()

        self.VoiceRecord()

        self.ui.pushButton.setEnabled(False)
        self.ui.pushButton.setText("正在分析音频")

        QApplication.processEvents()

        self.VoiceAnalyse()

        self.ui.pushButton.setEnabled(True)
        self.ui.pushButton.setText("开始")

