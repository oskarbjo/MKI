
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
echarge = 1.602e-19

def beamCurrentGauss(nB,tBL,tBS):
    
    
    try:
        [fPos,ySpecPos]=loadSpectrumFromFile(nB, tBL, tBS)
        print('Spectrum data available on local drive')
    except:
        print('Calculating beam current...')
    sigma = tBL/4
    # maximum frequency of interest
    fMax = 1/sigma
    # set sampling frequency more than 2*Fmax
    fS = 10*fMax
    
    # set time duration of the signal
    dt = 1/fS
    
    tMin = -5*sigma
    tMax = (nB-1)*tBS+5*sigma
    # nS = (tMax-tMin)*fS+1;
    
    t = np.arange(tMin,tMax,dt)
    
    y = 0*t
    for n in range(1,nB):
#        print(n)
        y = y + 1/(nB*np.sqrt(2*np.pi)*sigma)*np.exp(-(t-(n-1)*tBS)**2/(2*sigma**2))

    return [t,y]




def beamSpec(t,y):

    #sampling interval of the signal is the inverse of the sampling frequency
    #and also the maximum frequency that can be found from the FFT
    fS = 1/(t[2]-t[1])
    
    
    #set the length of the FFT signal
    nFFT = 2**(nextpow2(np.size(y))+5)
    
    #sampling interval of the FFT
    dF = fS/nFFT
    
    
    w = np.fft.fft(y,nFFT)
    fPos = np.arange(0,nFFT/2,1)*dF
    temp = np.abs(w)
    ySpecPos = temp[np.int_(np.arange(1,nFFT/2+1,1))]/fS
    
    return[fPos,ySpecPos]



def RFLosses(f0,iB,ySpecPos,fPos,impFile):
    
    [fImp,zImp] = importImpedanceData(impFile)
    
    fMin = fImp[1]
    fMax = fImp[-1]
    
    # vector of interpolating frequencies
    fInt = np.arange(fMin,fMax,f0)
    
    
    # calculate spectrum at interpolating frequencies
    y_temp = interpolate.interp1d(fPos,ySpecPos)
    YInt = y_temp(fInt)
    
    #calculate impedance at interpolating frequencies
    z_temp = interpolate.interp1d(fImp,zImp)
    zInt = z_temp(fInt)
    
    temp = np.multiply((np.square(YInt)),zInt)
    pLoss = 2*np.square(iB)*np.sum(temp[1:-1]) #should it be from second element? Don't include DC?
    
    return [pLoss, fImp, zImp]



def importImpedanceData(filepath):

    #'MKI_PostLS1_impedance.txt'
    file=open(filepath,'r')
    #file=open('MKI_PostLS1_impedance.txt','r')
    lines=file.readlines()
    
    freq=[]
    R=[]
    for n in lines:
    #    R.append(n.split()[0])
        a=n.split()
        if a != []:
            freq.append(np.float(a[0])*1e6)
            R.append(np.float(a[1]))
    file.close()
    
    return [freq,R]


def nextpow2(i):
    n = 1
    j=0
    while n < i: 
        n = n*2
        j = j+1
    return j


def loadSpectrumFromFile(nB,tBL,tBS):
    txt1 = str(tBL).replace('.','p')
    filepath = '//cernbox-smb.cern.ch/eos/user/o/objorkqv/Documents/Python/BEAM_SPECTRUM_DATA/'
    filename_freq = 'freq_beamSpectrum_' + str(nB) + 'bunches_' + txt1 + 'bunchLength_25nsbunchSeparation.dat'
    filename_spec = 'spec_beamSpectrum_' + str(nB) + 'bunches_' + txt1 + 'bunchLength_25nsbunchSeparation.dat'
    print('File found: ' + filepath+filename_freq)
    fPos = np.load(filepath + filename_freq,allow_pickle=True)
    ySpecPos = np.load(filepath + filename_spec,allow_pickle=True)
    return [fPos,ySpecPos]

def tBL_and_N_sweep(nB,tBS,f0,impFile):
    Npts = 21
    tBL_values = np.around(np.linspace(0.5e-9,1.5e-9,Npts), decimals=13)
    N_values = np.linspace(1e11,2e11,Npts)
    P_matrix = np.empty([Npts,Npts])
    p=0
    Nruns=Npts*Npts
    for i in range(0,Npts):
        fPos = []
        ySpecPos = []
        try: #If already present, load beam spectrum from file
            [fPos,ySpecPos]=loadSpectrumFromFile(nB,tBL_values[i],tBS)
            print('Beam spectrum file found')
        except: #Otherwise, compute beam spectrum
            print('File not found, computing beam spectrum')
            [t,y] = beamCurrentGauss(nB,tBL_values[i],tBS)
            [fPos,ySpecPos] = beamSpec(t,y)
        for j in range(0,Npts):
            iB = N_values[j]*nB*echarge*f0
            temp = RFLosses(f0,iB,ySpecPos,fPos,impFile)
            P_matrix[i][j] = temp[0]
            p=p+1
            print('Computing P matrix, run', p, ' of ', Nruns)
    return [P_matrix, tBL_values, N_values]


def tBL_sweep(N,nB,tBS,f0,impFile):
    Npts = 11
    tBL_values = np.around(np.linspace(1e-9,1.8e-9,Npts), decimals=13)
    P_array = np.empty([Npts,1])
    p=0
    echarge = 1.602e-19
    Nruns=Npts
    iB = N*nB*echarge*f0
    for i in range(0,Npts):
        fPos = []
        ySpecPos = []
        try: #If already present, load beam spectrum from file
            [fPos,ySpecPos]=loadSpectrumFromFile(nB,tBL_values[i],tBS)
            print('Beam spectrum file found')
        except: #Otherwise, compute beam spectrum
            print('File not found, computing beam spectrum')
            [t,y] = beamCurrentGauss(nB,tBL_values[i],tBS)
            [fPos,ySpecPos] = beamSpec(t,y)
        temp = RFLosses(f0,iB,ySpecPos,fPos,impFile)
        P_array[i] = temp[0]
        p=p+1
        print('Computing P matrix, run', p, ' of ', Nruns)
    return [P_array, tBL_values]


def nB_sweep(N,tBS,f0,impFile):
    Npts = 16
    nB = np.int_(np.linspace(1700,2300,Npts))
    tBL_values = 1.2e-9
    P_array = np.empty([Npts,1])
    p=0
    echarge = 1.602e-19
    Nruns=Npts
    iB = N*nB*echarge*f0
    for i in range(0,Npts):
        fPos = []
        ySpecPos = []
        try: #If already present, load beam spectrum from file
            [fPos,ySpecPos]=loadSpectrumFromFile(nB[i],tBL_values,tBS)
            print('Beam spectrum file found')
        except: #Otherwise, compute beam spectrum
            print('File not found, computing beam spectrum')
            [t,y] = beamCurrentGauss(nB[i],tBL_values,tBS)
            [fPos,ySpecPos] = beamSpec(t,y)
        temp = RFLosses(f0,iB[i],ySpecPos,fPos,impFile)
        P_array[i] = temp[0]
        p=p+1
        print('Computing P matrix, run', p, ' of ', Nruns)
    return [P_array, nB]

def N_sweep(nB,tBS,f0,impFile):
    Npts = 11
    tBL_values = 1.1e-9
    N = np.linspace(1.3e11,1.5e11,Npts)
    P_array = np.empty([Npts,1])
    p=0
    echarge = 1.602e-19
    Nruns=Npts
    iB = N*nB*echarge*f0
    for i in range(0,Npts):
        fPos = []
        ySpecPos = []
        try: #If already present, load beam spectrum from file
            [fPos,ySpecPos]=loadSpectrumFromFile(nB,tBL_values,tBS)
            print('Beam spectrum file found')
        except: #Otherwise, compute beam spectrum
            print('File not found, computing beam spectrum')
            [t,y] = beamCurrentGauss(nB[i],tBL_values,tBS)
            [fPos,ySpecPos] = beamSpec(t,y)
        temp = RFLosses(f0,iB[i],ySpecPos,fPos,impFile)
        P_array[i] = temp[0]
        p=p+1
        print('Computing P matrix, run', p, ' of ', Nruns)
    return [P_array, N]

def getTempFromP(inputPower,lossP,lossT):
    Tdata_TEMP = interpolate.interp1d(lossP,lossT)
    Tdata = Tdata_TEMP(inputPower)
    return Tdata

def T_vs_tBL_and_N(P_mat,P_losses,T_losses):
    #Temperature data
    # from 1D power deposition data
    # PLS1 stands for post-LS1 design
    #PdataPLS1 = [0, 35, 51, 127, 254, 508]
    #TdataPLS1Yoke = [22, 57, 73, 126, 213, 330]
    # Up stands for upgraded design
    #PdataUp = [0, 37, 92.5, 185, 370]
    #TdataUpRing = [22, 117, 212, 318, 462]
    #TdataUpYoke = [22, 63, 107, 154, 218]

    Npts = np.size(P_mat[0])
    T_mat = np.empty([Npts,Npts])
    Tdata_TEMP = interpolate.interp1d(P_losses,T_losses) #Default interpolation method = linear
    p=0
    Nruns = Npts*Npts
    for i in range(0,Npts):
        for j in range(0,Npts):
            try:
                T_mat[i][j] = Tdata_TEMP(P_mat[i][j])
            except:
                T_mat[i][j] = 0
            p=p+1
            print('Computing T matrix, run', p, ' of ', Nruns)
 
    return T_mat

def RFLosses_spect(f0,iB,ySpecPos,fPos,impFile):
    
    [fImp,zImp] = importImpedanceData(impFile)
    
    fMin = fImp[0]
    fMax = fImp[-1]
    
    # vector of interpolating frequencies
    fInt = np.arange(fMin,fMax,f0)
    
    
    # calculate spectrum at interpolating frequencies
    y_temp = interpolate.interp1d(fPos,ySpecPos)
    YInt = y_temp(fInt)
    
    #calculate impedance at interpolating frequencies
    z_temp = interpolate.interp1d(fImp,zImp)
    zInt = z_temp(fInt)
    
    temp = np.multiply((np.square(YInt)),zInt)
    pLoss = 2*np.square(iB)*np.sum(temp[0:]) #should it be from second element? Don't include DC?
    pLoss_spect = 2*np.square(iB)*temp[0:]
    spectSqr = (np.square(YInt))
    return [pLoss, fInt, pLoss_spect, spectSqr, zInt, YInt]



def findMostContributingf0(beamSpect,spectFreq, impFile, iB):
    f0 = 11245
    [pLoss, fInt, pLoss_spect, spectSqr, zInt, YInt] = RFLosses_spect(f0, iB, beamSpect, spectFreq, impFile)
    
    mostContributingf0_index = sorted(range(len(pLoss_spect)),key=lambda k: pLoss_spect[k]) #find indices of most contributing frequencies
    mostContributingf0_index = mostContributingf0_index[::-1]
    mostContributingf0 = fInt[mostContributingf0_index]
    
    return [mostContributingf0, mostContributingf0_index, fInt, pLoss_spect, spectSqr, zInt]
    
    
    
def import_CST_losses(filepath): # Don't remove header!
    positionMatrix = []
    dataMatrix = []
    
    file=open(filepath,'r')
    print('File found: ' + filepath)
    lines=file.readlines()
    lines = lines[2:] #skip header (2 rows)
    tempData = []
    for n in lines:
        a=n.split()
        if a != []:
            tempData.append(np.float(a[3]))
            positionMatrix.append([np.float(a[0]),np.float(a[1]),np.float(a[2])])
    tempData = np.array(tempData)
    dataMatrix.append(tempData)
    file.close()
    dataMatrix = np.array(dataMatrix) #Unit of dataMatrix is [W] (total power in each cell) and sum of all cells should be 1 W
    
    
    positionMatrix = np.asarray(positionMatrix)
    positionMatrix = positionMatrix.transpose()
    
    return [positionMatrix, dataMatrix]


def calculateAnalyticBeamSpectrum(Nbunches,Fmin,Fmax,Npts,sigma):
    
    analyticFreq = np.linspace(Fmin,Fmax,Npts) #Number of pts corresponds to what beamCurrentGauss function would
    mu = 2e-9
    pi = np.pi
    analyticFreqSqr = np.square(analyticFreq)
    analyticSpec = analyticFreq*0
    
    for i in range(0,Nbunches):
        analyticSpec = analyticSpec + (1/np.sqrt(2*pi))*np.exp(-(analyticFreqSqr * 2*pi * sigma**2)/2 + 1j*mu*2*pi*analyticFreq)
        mu = mu + 25e-9
        
    return[analyticFreq,analyticSpec]

def linearInterp(x1,x2,y1,y2,yTarget):
    #find linear equation y = kx + m
    k = (y2 - y1)/(x2 - x1)
    m = y2 - k*x2
    x = (yTarget - m)/k
    return x

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