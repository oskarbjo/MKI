
import functions
import numpy as np
import matplotlib.pyplot as plt
#This code does not work - needs better signal processing
def main():
    path = r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKI cool\MKI cool resonant measurements\MKICOOL_SNP.S2P"
    getQualityFactor(path)


def getQualityFactor(path):
    
    S = functions.SNPfile(path)
    [localMaxInd,localMaxVal]=findLocalMax(S.S21lin)
    [ind,f,Q,S21] = find3dBpoints(localMaxInd,localMaxVal,S)
    
    writeQtoFile(f,Q,S21)
    
    plotPts(f,S)
    
def find3dBpoints(localMaxInd,localMaxVal,S):
    ind = []
    Q = []
    f = []
    S21 = []
    for i in range(0,len(localMaxVal)):
        leftind = 0
        rightind = 0
        try:
            j=0
            indCenter = localMaxInd[i]
            while(S.S21lin[indCenter-j] > (S.S21lin[indCenter]/np.sqrt(2)) and indCenter-j>-1):
                j=j+1
            leftind = indCenter - j
            j=0
            while(S.S21lin[indCenter+j] > (S.S21lin[indCenter]/np.sqrt(2)) and S.S21lin[indCenter+j] < S.S21lin[indCenter]):
                j=j+1
            rightind = indCenter + j
            ind.append(indCenter) #indCenter is true resonant peak so add to list of resonance indices
            Q.append(calculateQ(leftind,rightind,indCenter,S))
            f.append(S.freq[indCenter])
            S21.append(S.S21db[indCenter])
        except:
            print('Not resonant peak')
        
        
    return[ind,f,Q,S21]


def calculateQ(leftind,rightind,indCenter,S):
    
    f1 = functions.linearInterp(S.freq[leftind-1], S.freq[leftind], S.S21lin[leftind-1], S.S21lin[leftind], S.S21lin[indCenter]/np.sqrt(2))
    f2 = functions.linearInterp(S.freq[rightind], S.freq[rightind+1], S.S21lin[rightind], S.S21lin[rightind+1], S.S21lin[indCenter]/np.sqrt(2))
    Q = S.freq[indCenter]/(f2-f1)
    return Q

def findLocalMax(S21lin):
    localMaxTempInd = []
    localMaxTemp = []
    for i in range(1,len(S21lin)-1):
        if (S21lin[i-1] < S21lin[i] and S21lin[i-2] < S21lin[i-1] and S21lin[i-3] < S21lin[i-2] and S21lin[i-4] < S21lin[i-3]
            and S21lin[i+1] < S21lin[i] and S21lin[i+2] < S21lin[i+1] and S21lin[i+3] < S21lin[i+1] and S21lin[i+4] < S21lin[i+3]):
            localMaxTemp.append(S21lin[i])
            localMaxTempInd.append(i)
    return [localMaxTempInd,localMaxTemp]


def writeQtoFile(f,Q,S21):
    fileName = r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKI cool\MKI cool resonant measurements/testfile.txt"
    file = open(fileName,"w")
    for i in range(0,len(Q)):
        line = str(f[i]) + ',' + str(Q[i]) + ',' +  str(S21[i]) + '\n'
        file.write(line)    
        

def plotPts(f,S):
    plt.figure()
    plt.plot(S.freq,S.S21db)
    plt.scatter(f,np.zeros([len(f),1])-30)
    plt.show()

if __name__ == "__main__":
    main()
    
