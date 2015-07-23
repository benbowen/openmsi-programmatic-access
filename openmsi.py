import json
import numpy as np
import getpass
import re
from elements import ELEMENTS
import copy

def authenticateUser(client,username):
	password = getpass.getpass()
	authURL = 'https://openmsi.nersc.gov/openmsi/client/login/'
	# Retrieve the CSRF token first
	client.get(authURL)  # sets cookie
	csrftoken = client.cookies['csrftoken']
	login_data = dict(username=username, password=password, csrfmiddlewaretoken=csrftoken, next='/')
	r = client.post(authURL, data=login_data, headers=dict(Referer=authURL))
	return client
	
def getFilelist(client):
	payload = {'format':'JSON','mtype':'filelistView'}
	url = 'https://openmsi.nersc.gov/openmsi/qmetadata'
	r = client.get(url,params=payload)
	fileList = json.loads(r.content)
	return fileList.keys()

def getMZ(client,filename,expIndex,dataIndex):
	payload = {'file':filename,
          'expIndex':expIndex,'dataIndex':dataIndex,'qspectrum_viewerOption':'0',
          'qslice_viewerOption':'0',
          'col':0,'row':0,
          'findPeak':'0','format':'JSON'}
	url = 'https://openmsi.nersc.gov/openmsi/qmz'
	r = client.get(url,params=payload)
	data = json.loads(r.content)
	return np.asarray(data[u'values_spectra'])

def chemformula_struct(formula):
	matEle = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
	for idx, row in enumerate(matEle):
	    id, num = row
	    try:
	    	float(num)
	    except:
	  		matEle[idx] = (id, '1')
	return matEle

def monoisotopicmass(formula):
	f = chemformula_struct(formula)
	m = np.zeros((len(f)))
	for i in range(len(f)):
	    e = ELEMENTS[f[i][0]]
	    maxIso = 0
	    for iso in e.isotopes:
	        if e.isotopes[iso].abundance > maxIso:
	            maxIso = e.isotopes[iso].abundance
	            m[i] = float(e.isotopes[iso].mass) * float(f[i][1])
	return np.sum(m)

def chemformula_list(formula):
	f = chemformula_struct(formula)
	str = 'HCNOSPDX'
	fList = np.zeros((len(str)))
	for i in range(len(f)):
	    fList[str.index(f[i][0])] = f[i][1]
	return fList

def isotopic_pattern(fList,DAbund, num_D_sites):
	# fList is a list of formula vectors

	import numpy as np
	#loop through all the formulae to get the unique mz list
	mzvec = []
	for iii,fVec in enumerate(fList):
		x1,y1,m1 = isotope(fVec,DAbund)
		mzvec = np.union1d(mzvec,x1)
		labVec = fVec[:]
		labVec[6] = num_D_sites[iii]
		labVec[0] = fVec[0] - num_D_sites[iii]
		x2,y2,m2 = isotope(labVec,DAbund)
		mzvec = np.union1d(mzvec,x2)
	mzvec = np.round(mzvec*200)/200 #hack to round for unique
	mzvec = np.unique(mzvec)
	for i in range(len(mzvec)-1):
	    if abs(mzvec[i]-mzvec[i+1])<0.1: #hack to group less than 0.1. should be parameter
	        mzvec[i+1] = mzvec[i]
	            
	mzvec = np.unique(mzvec)

	#go back through the formulae to put the intensity values onto the mz list
	isoY = np.zeros((len(fList)*2,len(mzvec)))
	for iii,fVec in enumerate(fList):
		x1,y1,m1 = isotope(fVec,DAbund)
		for i,x in enumerate(x1):
		    idx = np.argmin(abs(x-mzvec))
		    isoY[iii*2+0,idx] = y1[i]

		labVec = copy.deepcopy(fVec)
		labVec[6] = num_D_sites[iii]
		labVec[0] = fVec[0] - num_D_sites[iii]
		x2,y2,m2 = isotope(labVec,DAbund)
		for i,x in enumerate(x2):
		    idx = np.argmin(abs(x-mzvec))
		    isoY[iii*2+1,idx] = y2[i]

	return mzvec, isoY

def isotope(fVec,DAbund):
	'''
	%
	% Calculates isotopic distributions including isotopic fine structure
	% of molecules using FFT and various scaling 'tricks'. Easily adopted
	% to molecules of any elemental composition (by altering MAX_ELEMENTS
	% and the nuclide matrix A). To simulate spectra, convolute with peak
	% shape using FFT.
	%
	% (C) 1999 by Magnus Palmblad, Division of Ion Physics, Uppsala Univ.
	% Acknowledgements:
	% Lars Larsson-Cohn, Dept. of Mathematical Statistics, Uppsala Univ.,
	% for help on theory of convolutions and FFT.
	% Jan Axelsson, Div. of Ion Physics, Uppsala Univ. for comments and ideas
	%
	% Contact Magnus Palmblad at magnus.palmblad@gmail.com if you should
	% have any questions or comments.
	%

	Converted to Python 1/10/08 by
	Brian H. Clowers bhclowers@gmail.com

	October 31, 2014
	Added Phosphorous and chemical formula parsing
	Added conditional specification of stable isotope composition
	Ben Bowen, ben.bowen@gmail.com

	fVec is a vector representing the chemical formula including deuterium
	# [H, C, N, O, S, P, D]
	DAbund is the amount of deuterium [0-1], 0.05 is typical
	'''
	import numpy as np
	import numpy.fft.fftpack as F
	# import time
	# import pylab as P


	def next2pow(x):
	    return 2**int(np.ceil(np.log(float(x))/np.log(2.0)))

	scaleFactor = 100000
	MAX_ELEMENTS=7+1  # add 1 due to mass correction 'element'
	MAX_ISOTOPES=4    # maxiumum # of isotopes for one element
	CUTOFF=1e-4    # relative intensity cutoff for plotting

	WINDOW_SIZE = 500
	#WINDOW_SIZE=input('Window size (in Da) ---> ');

	#RESOLUTION=input('Resolution (in Da) ----> ');  % mass unit used in vectors
	RESOLUTION = 0.5
	if RESOLUTION < 0.00001:#  % minimal mass step allowed
	  RESOLUTION = 0.00001
	elif RESOLUTION > 0.5:  # maximal mass step allowed
	  RESOLUTION = 0.5

	R=0.00001/RESOLUTION#  % R is used to scale nuclide masses (see below)

	WINDOW_SIZE=WINDOW_SIZE/RESOLUTION;   # convert window size to new mass units
	WINDOW_SIZE=next2pow(WINDOW_SIZE);  # fast radix-2 fast-Fourier transform algorithm

	if WINDOW_SIZE < np.round(496708*R)+1:
	  WINDOW_SIZE = nextpow2(np.round(496708*R)+1)  # just to make sure window is big enough

	# print 'Vector size: 1x%d'%WINDOW_SIZE

	#H378 C254 N65 O75 S6
	
	# M=np.array([378,254,65,75,6,0]) #% empiric formula, e.g. bovine insulin
	M=np.array(fVec) #% empiric formula, e.g. bovine insulin

	# isotopic abundances stored in matrix A (one row for each element)
	A=np.zeros((MAX_ELEMENTS,MAX_ISOTOPES,2));

	A[0][0,:] = [100783,0.9998443]#                 % 1H
	A[0][1,:] = [201410,0.0001557]#                 % 2H
	A[1][0,:] = [100000,0.98889]#                   % 12C
	A[1][1,:] = [200336,0.01111]#                   % 13C
	A[2][0,:] = [100307,0.99634]#                   % 14N
	A[2][1,:] = [200011,0.00366]#                   % 15N
	A[3][0,:] = [99492,0.997628]#                  % 16O
	A[3][1,:] = [199913,0.000372]#                  % 17O
	A[3][2,:] = [299916,0.002000]#                  % 18O
	A[4][0,:] = [97207,0.95018]#                   % 32S
	A[4][1,:] = [197146,0.00750]#                   % 33S
	A[4][2,:] = [296787,0.04215]#                   % 34S
	A[4][3,:] = [496708,0.00017]#                   % 36S
	A[5][0,:] = [97376,1.0]# Phosphorous
	A[6][0,:] = [100783,1.0-DAbund]#                 % 1H
	A[6][1,:] = [201410,DAbund]#                 % 2H
	A[7][0,:] = [100000,1.00000]#                   % for shifting mass so that Mmi is
	#                                             % near left limit of window
	mass_removed_vec = [0,11,13,15,31,30,0,-1]
	monoisotopic = 0.0
	for i,e in enumerate(fVec):
		monoisotopic = monoisotopic + ( (mass_removed_vec[i]*scaleFactor+A[i][0,0])*e / scaleFactor)

	Mmi=np.array([np.round(100783*R), np.round(100000*R),\
	             np.round(100307*R), np.round(99492*R), np.round(97207*R), np.round(97376*R), np.round(100783*R), 0])*M#  % (Virtual) monoisotopic mass in new units
	Mmi = Mmi.sum()
	#% mass shift so Mmi is in left limit of window:
	#print "Mmi",Mmi
	#print "Window", WINDOW_SIZE
	FOLDED=np.floor(Mmi/(WINDOW_SIZE-1))+1#  % folded FOLDED times (always one folding due to shift below)

	#% shift distribution to 1 Da from lower window limit:
	M[MAX_ELEMENTS-1]=np.ceil(((WINDOW_SIZE-1)-np.mod(Mmi,WINDOW_SIZE-1)+np.round(100000*R))*RESOLUTION)
	MASS_REMOVED=np.array(mass_removed_vec)*M#';  % correction for 'virtual' elements and mass shift
	MASS_REMOVED = MASS_REMOVED.sum()

	ptA=np.ones(WINDOW_SIZE);
	t_fft=0
	t_mult=0

	for i in xrange(MAX_ELEMENTS):
	    tA=np.zeros(WINDOW_SIZE)
	    for j in xrange(MAX_ISOTOPES):
	        if A[i][j,0] != 0:
	            tA[np.round(A[i][j,0]*R)]=A[i][j,1]#;  % put isotopic distribution in tA

	    tA=F.fft(tA) # FFT along elements isotopic distribution  O(nlogn)
	    tA=tA**M[i]#  % O(n)
	    ptA = ptA*tA#  % O(n)#this is where it is messing UP

	ptA=F.ifft(ptA).real#;  % O(nlogn)

	start = (FOLDED*(WINDOW_SIZE-1)+1)*RESOLUTION+MASS_REMOVED,(FOLDED+1)*(WINDOW_SIZE-1)*RESOLUTION+MASS_REMOVED
	stop = WINDOW_SIZE - 1

	MA=np.linspace((FOLDED*(WINDOW_SIZE-1)+1)*RESOLUTION+MASS_REMOVED,(FOLDED+1)*(WINDOW_SIZE-1)*RESOLUTION+MASS_REMOVED, WINDOW_SIZE-1)

	ind=np.where(ptA>CUTOFF)[0]

	x = MA[ind]
	y = ptA[ind]

	for i,xi in enumerate(x):
		x[i] = monoisotopic + (i*1.003355)


	return x,y,monoisotopic
