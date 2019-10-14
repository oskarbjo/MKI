
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import functions
from os import listdir
from os.path import isfile, join

impFile=r"\\cernbox-smb.cern.ch\eos\user\o\objorkqv\Documents\CST_EOS\mkiPostLS1_2000MHz_results_mostContributingf0\mkiPostLS1_imp.txt" #Remember to remove impedance file header
#impFile=r'C:/Users/objorkqv/eclipse-workspace/mki_powerloss/MKI_upgraded_impedance.txt'
#impFile=r'C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data/MKI_cool_impedance.txt'
#impFile=r'E:\CST\MKIcool\MKIcool remeshed\15mm insert_IMPEDANCE SIMULATION/MKIcool_impedance_REMESHED.txt'
# impFile=r'E:\CST\MKIcool\MKIcool remeshed\Reduced MKI cool 5/reduced_mkicool_5_impedance_model.txt'
#impFile=r'E:\CST\MKIcool\MKIcool remeshed\Damper optimization\simulated files\10mm/MKIcool_10mm_impedance.txt'
#impFile=r'E:\CST\MKIcool\MKIcool remeshed\Insert design suggestion/insert_design1_impedance.txt'
#impFile = r'E:\CST\MKIcool\MKIcool remeshed\Damper optimization\simulated files\0mm\MKIcool_0mm_inset_imp.txt'

spec_freq = np.load(r"C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data/spectrum_frequencies_2748bunches_1e-9tbl_1.15e11Int_25nsBS.dat",allow_pickle=True)
spec = np.load(r"C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data/spectrum_data_2748bunches_1e-9tbl_1.15e11Int_25nsBS.dat",allow_pickle=True)
frequencies = np.linspace(40,600,15)
N = 2.2e11
nB = 2700
echarge=1.602e-19
f0=11245
iB = N*nB*echarge*f0

def writeLossDataToFile(positionMatrix,totalDataMatrix,fileName):
	file = open(fileName,"w")
	for i in range(0,len(totalDataMatrix)):
		line = str(positionMatrix[0,i]) + ',' + str(positionMatrix[1,i]) + ',' +  str(positionMatrix[2,i]) + ',' + str(totalDataMatrix[i,0]) + '\n'
		file.write(line)

def checkMostContrf0():	
	[mostContributingf0, indices, fInt, pLoss_spect, spectSqr, zInt] = functions.findMostContributingf0(spec,spec_freq, impFile, iB)
	
	plt.figure()
	plt.plot(fInt,pLoss_spect)
	plt.plot(fInt,6*spectSqr/np.max(spectSqr))
	plt.plot(fInt,6*zInt/np.max(zInt))
	plt.show()
	
	
	for i in range(0,60):
		printThis = mostContributingf0/1e6
		print(i, '. ', printThis[i])
		

def calculatePHat(positionMatrix,dataMatrix): #Phat is the normalized power in the entire body at a certain freq
	#dV = (positionMatrix[0,1]-positionMatrix[0,0])*0.001
	pHat = np.zeros([np.size(dataMatrix[:,0]),np.size(dataMatrix[0,:])])
	for i in range(0,np.size(dataMatrix[:,0])):
				pHat[i,:] = dataMatrix[i,:] / np.sum(dataMatrix[i,:])
	
	return pHat


def calculateAlphaN(spec_freq, spec, impFile, interpFreq):
	[fImp,zImp] = functions.importImpedanceData(impFile)
	# calculate spectrum at interpolating frequencies
	y_temp = interpolate.interp1d(spec_freq,spec)
	YInt = y_temp(interpFreq)
	#calculate impedance at interpolating frequencies
	z_temp = interpolate.interp1d(fImp,zImp)
	zInt = z_temp(interpFreq)
	wp = np.multiply((np.square(YInt)),zInt)
	alphaN = np.zeros([len(wp),1])
	for i in range(0,len(interpFreq)):
		alphaN[i] = wp[i]/np.sum(wp)
	return alphaN
	

def calculatePLoss(pHat,alphaN):
	
	Ptot = 1 #Specify the total power losses
	PLoss = np.zeros([np.size(pHat[0,:])])
	temp = PLoss
	for i in range(0,len(alphaN)):
		temp = Ptot*pHat[i,:]*alphaN[i]
		PLoss = PLoss + temp
	
	return PLoss

def sortFileNamesByFreq(filenames):
	fileFreq=np.zeros([1,len(filenames)])
	start = 'f='
	end = ')'
	for i in range(0,len(filenames)):
		filename = filenames[i]
		fileFreq[0,i]=np.double(filename[filename.find(start)+len(start):filename.rfind(end)])
	ind=np.argsort(fileFreq)
	sortedFileNames = []
	for i in range(0,len(filenames)):
		print(fileFreq[0,ind[0,i]])
		sortedFileNames = sortedFileNames + [filenames[ind[0,i]]]
	return sortedFileNames

#freq = np.load(r'C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data/mostContributingf0_cool_ASCENDING.dat',allow_pickle=True)
freq = np.load(r"C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data\mostContrFreqAsc_MKIPostLS1.npy")
totalDataMatrix = []
alphaN = calculateAlphaN(spec_freq, spec, impFile, freq) #Array containing all scaling factors
path = r'\\cernbox-smb.cern.ch\eos\user\o\objorkqv\Documents\CST_EOS\mkiPostLS1_2000MHz_results_mostContributingf0\mkiPostLS1\Export\3d/'
onlyfiles = sortFileNamesByFreq([f for f in listdir(path) if isfile(join(path, f))])
i=0
for filename in onlyfiles:
	filePath = path + filename
	print('File: ' + filename + ', freq = ' + str(freq[i]))
	[positionMatrix, dataMatrix] = functions.import_CST_losses(filePath)
	if i == 0:
		totalDataMatrix = np.zeros(np.shape(dataMatrix))
	pHat = calculatePHat(positionMatrix,dataMatrix) #Normalized power distribution so that its sum is 1
	PLoss = pHat*alphaN[i]
	i = i+1
	totalDataMatrix = totalDataMatrix + PLoss
	#PLoss = np.transpose(dataMatrix/(1e-6*np.max(dataMatrix)))
dV = np.power(0.001*(positionMatrix[0,1]-positionMatrix[0,0]),3) #volume element
totalDataMatrix = totalDataMatrix/dV #Make unit W/m3
totalDataMatrix = np.transpose(totalDataMatrix)
writeLossDataToFile(positionMatrix,totalDataMatrix,r'C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\data\MKIcool_losses_ansys/MKIcool_xxx.txt')



