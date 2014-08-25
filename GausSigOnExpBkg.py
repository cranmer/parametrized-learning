'''
Define model mu_s*Gaus(x|alpha,sigma)+mu_b*flat(x)
Generate {x} for several {alpha}
Calculate power (expected significance) for some alpha using profile likelihood approach
Train NN for alpha=0. 
	make (histfactory/on the fly) model for NN with alpha variations
		- calculate power
Train NN with {x,alpha}
	a) make (histfactory/on the fly) model for NN with alpha variations using same alpha as input to NN
		- calculate power
	b) make (histfactory/on the fly) model for NN with alpha variations maximizing NN w.r.t. alpha
		- calculate power
	d) instead of maximizing per event, maximize sum NN and build corresponding distribution
		- note, this is variant to non-linearchanges in NN -> NN' so seems like bad idea
'''


import ROOT
import numpy as np
from sklearn import svm, linear_model, gaussian_process
import matplotlib.pyplot as plt

def makeData():
	musteps=10
	numTrain=500
	numTest=numTrain

	w = ROOT.RooWorkspace('w')
	w.factory('Gaussian::g(x[-5,5],mu[0,-3,3],sigma[0.5, 0, 2])')
	w.factory('Exponential::e(x,tau[-.15,-3,0])')
	w.factory('SUM::model(s[50,0,100]*g,b[100,0,1000]*e)')
	w.Print() #this isn't displaying in iPython

	x = w.var('x')
	mu = w.var('mu')
	pdf = w.pdf('model')
	sigpdf = w.pdf('g')
	bkgpdf = w.pdf('e')
	frame = x.frame()

	traindata = np.zeros((2*numTrain*musteps,2))
	targetdata = np.zeros(2*numTrain*musteps)

	testdata = np.zeros((numTest*musteps,2))
	testdata1 = np.zeros((numTest*musteps,2))

	for mustep, muval in enumerate(np.linspace(-1,1,musteps)):
		mu.setVal(muval)
		sigdata = sigpdf.generate(ROOT.RooArgSet(x),numTrain)
		bkgdata = bkgpdf.generate(ROOT.RooArgSet(x),numTrain)
		alldata = pdf.generate(ROOT.RooArgSet(x),numTest)



		for i in range(numTrain):
			traindata[ i+mustep*numTrain,0] = sigdata.get(i).getRealValue('x')
			traindata[ i+mustep*numTrain,1] = muval
			targetdata[i+mustep*numTrain] = 1
		for i in range(numTrain):
			traindata[ i+mustep*numTrain+musteps*numTrain,0] = bkgdata.get(i).getRealValue('x')
			traindata[ i+mustep*numTrain+musteps*numTrain,1] = muval
			targetdata[i+mustep*numTrain+musteps*numTrain] = 0
		for i in range(numTest):
			testdata[i+mustep*numTest,0] = alldata.get(i).getRealValue('x')
			testdata[i+mustep*numTest,1] = 0.

		for i in range(numTest):
			testdata1[i+mustep*numTest,0] = alldata.get(i).getRealValue('x')
			testdata1[i+mustep*numTest,1] = 2*(i%2)-1.

	#print train and test data to file
	f = open('traindata.dat','w')
	for i, x in enumerate(traindata):
		f.write('{x} {mass} {target}\n'.format(x=x[0], mass=x[1], target=targetdata[i]))
	f.close()

	f = open('testdata.dat','w')
	for i, x in enumerate(testdata):
		f.write('{x} {mass}\n'.format(x=x[0], mass=x[1]))
	f.close()

	# do regression with nuisance parameter input
	clf = svm.NuSVR()
	clf.fit(traindata, targetdata)  

	# evaluate with different asssumed mass values
	outputs=clf.predict(testdata)
	outputs1=clf.predict(testdata1)

	# do regression without
	clf = svm.NuSVR()
	temp = traindata[:,0].reshape(len(traindata),1)
	clf.fit(temp, targetdata)  
	outputsSmeared=clf.predict(testdata1[:,0].reshape(len(testdata1),1) )

	# make scatter plot of regression vs. likelihood ratio
	x = w.var('x')
	mu.setVal(0)
	LRs=np.zeros(len(testdata))
	for i, xx in enumerate(testdata):
		x.setVal(xx[0])
		LRs[i] = sigpdf.getVal(ROOT.RooArgSet(x))/bkgpdf.getVal(ROOT.RooArgSet(x))
	LRs = LRs / np.max(LRs) 


	#plt.scatter(traindata[:,0],traindata[:,1],color='green')

	plt.scatter(traindata[:,0],targetdata,color='red')
	plt.scatter(testdata[:,0],outputs,color='green')
	plt.scatter(testdata1[:,0],outputs1,color='purple')
	plt.scatter(testdata1[:,0],outputsSmeared,color='orange')
	plt.scatter(testdata[:,0],LRs,color='black')

	#plt.show()
	plt.savefig('example.pdf')


if __name__ == '__main__':
	makeData()


	'''	
	#write ttrees, some issue with ownership?
	x = w.var('x')
	rootfile = ROOT.TFile('sigdata.root','RECREATE')
	tempdata = ROOT.RooDataSet('sigdata','sigdata',ROOT.RooArgSet(x))
	tempdata.setDefaultStorageType(0) # tree
	tempdata.append(sigdata)
	tree = tempdata.store().tree()
	tree.SetDirectory(ROOT.gDirectory.pwd())
	tempdata = ROOT.RooDataSet('bkgdata','bkgdata',ROOT.RooArgSet(x))
	tempdata.setDefaultStorageType(0) # tree
	tempdata.append(bkgdata)
	tree = tempdata.store().tree()
	tree.SetDirectory(ROOT.gDirectory.pwd())
	#tree.SetDirectory(ROOT.gDirectory)
	print tree
	print tree.GetDirectory()
	#tree.Write()
	rootfile.Write()
	rootfile.Close()
	'''
