# This Python file uses the following encoding: utf-8
import sys
from PySide2.QtWidgets import QApplication, QMainWindow
from mainwindow import *
import voicerecognition




if __name__ == "__main__":

    if len(sys.argv)==1:
        app = QApplication([])
        window = MainWindow()
        window.show()
        window.InitVoiceHandle()
        sys.exit(app.exec_())
    elif sys.argv[1]=="--NoGUI":
        voicerecognition.VoiceRecognitionNoGUI()
    
    elif sys.argv[1]=="--version":
        print("V1.0")
        print("Powered by ZZS 2019©All Rights Reserved")
        print("本程序由160200531周子顺制作，欢迎使用")
    
    elif sys.argv[1]=="--help":
        print("这是一个可以识别0-9的语音识别小程序")
        print("用法:")
        print("    --NoGUI    不启用图形化界面")
        print("    --help     查看帮助")
        print("    --version  查看版本")
    
    else:
        print("这是一个可以识别0-9的语音识别小程序")
        print("用法:")
        print("    --NoGUI    不启用图形化界面")
        print("    --help     查看帮助")
        print("    --version  查看版本")

