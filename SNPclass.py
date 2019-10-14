import numpy as np
import matplotlib.pyplot as plt
path = r'C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data/HELIAXWBPMSY4L5.s2p'

class SNPfile:
    def __init__(self,filePath):
        self.S11lin = []
        self.S21lin = []
        self.S12lin = []
        self.S22lin = []
        self.freq = []
        self.S11db = []
        self.S21db = []
        self.S12db = []
        self.S22db = []
        self.S11angle = []
        self.S21angle = []
        self.S12angle = []
        self.S22angle = []
        path = filePath
        self.extractSParamData(filePath)
        
    def extractSParamData(self,filePath):
        file=open(filePath,'r')
        lines=file.readlines()
        for i in range(0,len(lines)):
            data = self.separateParameters(lines[i])
            try:
                self.freq = self.freq + [np.double(data[0])]
                self.S11db = self.S11db + [np.double(data[1])]
                self.S21db = self.S21db + [np.double(data[3])]
                self.S12db = self.S12db + [np.double(data[5])]
                self.S22db = self.S22db + [np.double(data[7])]
                self.S11lin = self.S11lin + [self.dBtoLin(np.double(data[1]))]
                self.S21lin = self.S21lin + [self.dBtoLin(np.double(data[3]))]
                self.S12lin = self.S12lin + [self.dBtoLin(np.double(data[5]))]
                self.S22lin = self.S22lin + [self.dBtoLin(np.double(data[7]))]
                self.S11angle = self.S11angle + [np.double(data[2])]
                self.S21angle = self.S21angle + [np.double(data[4])]
                self.S12angle = self.S12angle + [np.double(data[6])]
                self.S22angle = self.S22angle + [np.double(data[8])]
            except:
                print('Removing SNP header')
        
        
    def separateParameters(self,line):
        line=line.strip('\n')
        line=line.split('\t')
        return line
    
    def plotParam(self,Sparam):
        plt.figure()
        plt.plot(self.freq,Sparam)
        plt.show()
    
    def dBtoLin(self,dBdata):
        linData = np.power(10,dBdata/20)
        return linData
        
        
def main():
    Sparam = SNPfile(path)
    Sparam.plotParam(Sparam.S11db)
    

if __name__ == "__main__":
    main()
    
    