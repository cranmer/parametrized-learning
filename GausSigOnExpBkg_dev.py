'''
author Kyle Cranmer <kyle.cranmer@nyu.edu>

Define model mu_s*Gaus(x|alpha,sigma)+mu_b*flat(x)
Generate {x} for several {alpha}
Calculate power (expected significance) for some alpha using profile likelihood approach
1) Train NN for alpha=0. 
	1a) make (histfactory/on the fly) model for NN with alpha variations
		- calculate power
	1b) make pdf on the fly for each value of alpha
2) Train NN with {x,alpha}
	a) make histfactory model for NN with alpha variations using same alpha as input to NN
		- calculate power
	b) make pdf on the fly for NN with alpha variations using same alpha as input to NN
		- calculate power
'''


import ROOT
import numpy as np
from sklearn import svm, linear_model, gaussian_process
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

import matplotlib.pyplot as plt

def trainAndTest(nmax=-1):
	print "Entering trainAndTest"
	trainAndTarget = np.loadtxt('traindata.dat')
	#to use nmax, need to shuffle first
	traindata = trainAndTarget[:nmax,0:2]
	targetdata = trainAndTarget[:nmax,2]

	testdata = np.loadtxt('testdata.dat')
	testdata1 = np.loadtxt('testdata1.dat')

	#transform data

	# do regression with nuisance parameter input
	clf = svm.NuSVR()
	'''
	# attempt at using Gaussian Processes as in 
	# http://scikit-learn.org/stable/auto_examples/gaussian_process/\
	# plot_gp_probabilistic_classification_after_regression.html
	clf = gaussian_process.GaussianProcess(theta0=1)
	noise = np.random.normal(0,.1,len(traindata))
	dummy=np.zeros(len(traindata))
	offset = np.column_stack((dummy,noise))
	targetdata= targetdata+noise
	traindata=traindata+offset
	'''
   
	clf.fit(traindata, targetdata)  


	# evaluate with different asssumed mass values
	outputs=clf.predict(testdata)
	outputs1=clf.predict(testdata1)

	'''
	# Create and fit an AdaBoosted decision tree
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), 
		algorithm="SAMME", n_estimators=200)
	bdt.fit(traindata, targetdata)
	outputs = bdt.decision_function(testdata)
	outputs = (outputs-outputs.min())/(outputs.max()-outputs.min())
	print outputs
	outputs1 = bdt.decision_function(testdata1)
	'''

	'''	
	# an unsupervised method, not using properly here.
	brbm = BernoulliRBM()
	brbm.fit(traindata,targetdata)
	outputs = brbm.transform(testdata)
	print np.shape(outputs)
	'''

	# make scatter plot of regression vs. likelihood ratio
	f = ROOT.TFile('workspace_GausSigOnExpBkg.root','r')
	w = f.Get('w')
	x = w.var('x')
	mu = w.var('mu')
	sigpdf = w.pdf('g')
	bkgpdf = w.pdf('e')

	mu.setVal(0)
	LRs=np.zeros(len(testdata))
	for i, xx in enumerate(testdata):
		x.setVal(xx[0])
		LRs[i] = sigpdf.getVal(ROOT.RooArgSet(x))/bkgpdf.getVal(ROOT.RooArgSet(x))
	LRs = LRs / np.max(LRs) 

	plt.scatter(traindata[:,0],targetdata,color='red')
	plt.scatter(testdata[:,0],outputs,color='green')
	#plt.scatter(testdata1[:,0],outputs1,color='purple')
	plt.scatter(testdata[:,0],LRs,color='black')

	plt.savefig('example.pdf')
	plt.show()

	#make histograms of output for signal and background samples
	#FIX: currently using the trainig data b/c test data has no labels
	sigoutputs=clf.predict(traindata[:len(traindata)/2])
	bkgoutputs=clf.predict(traindata[len(traindata)/2:])
	plt.hist(sigoutputs, alpha=.5)
	plt.hist(bkgoutputs, alpha=.5)
	plt.show()

	#nicer plot example: http://scikit-learn.org/stable/auto_examples/svm/plot_svm_nonlinear.html




def makeMomentMorph(w,interpParam, observable, pdfList, paramPoints):
	paramVec = ROOT.TVectorD(len(paramPoints))
	for i, p in enumerate(paramPoints):
		paramVec[i]=p #seems silly, but other constructor gave problems

	pdfs = ROOT.RooArgList()
	for pdf in pdfList:
		pdfs.add(pdf)

	setting = ROOT.RooMomentMorph.Linear
	morph = ROOT.RooMomentMorph("morph",'morph',interpParam,
		ROOT.RooArgList(observable),pdfs, paramVec,setting)
	morph.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel("RooBinIntegrator")

	#getattr(w,'import')(morph) # work around for morph = w.import(morph)
	getattr(w,'import')(ROOT.RooArgSet(morph),ROOT.RooFit.RecycleConflictNodes()) # work around for morph = w.import(morph)
	#getattr(w,'import')(morph,ROOT.RooFit.RecycleConflictNodes()) # work around for morph = w.import(morph)
	#getattr(w,'import')(morph,ROOT.RooFit.RenameConflictNodes("clash")) # work around for morph = w.import(morph)
	#getattr(w,'import')(morph,False) # treats like a TObject, looses name?
	w.Print()

	return w

def makeBSpline(w,interpParam, observable, pdfList, paramPoints):
	ROOT.gROOT.ProcessLine(".L RooBSpline.cxx+")

	paramVec = ROOT.TVectorD(len(paramPoints))
	tValues = ROOT.std.vector("double")()
	for i, p in enumerate(paramPoints):
		paramVec[i]=p #seems silly, but other constructor gave problems
		tValues.push_back(p)

	order=3
	bspb = ROOT.RooStats.HistFactory.RooBSplineBases( "bases", "bases", order, tValues, interpParam )

	pdfs = ROOT.RooArgList()
	for pdf in pdfList:
		pdfs.add(pdf)

	#this makes a function
	morphfunc = ROOT.RooStats.HistFactory.RooBSpline( "morphf", "morphf", pdfs, bspb, ROOT.RooArgSet() )

	#if you want to convert it into a PDF
	morph = ROOT.RooRealSumPdf('morph','morph', ROOT.RooArgList(morphfunc), ROOT.RooArgList())

	print morph
	#getattr(w,'import')(morph) # work around for morph = w.import(morph)
	getattr(w,'import')(ROOT.RooArgSet(morph),ROOT.RooFit.RecycleConflictNodes()) # work around for morph = w.import(morph)

	return w

def makePdf():
	# making plots for when mu is and isn't included in training
	# using data with several mu values
	# FIX: use different test/train samples (not a big deal for simple problem)

	print "Entering trainAndTest"
	trainAndTarget = np.loadtxt('traindata.dat')
	#to use nmax, need to shuffle first
	traindata = trainAndTarget[:,0:2]
	targetdata = trainAndTarget[:,2]

	massPoints = np.unique(traindata[:,1])
	chunk = len(traindata)/len(massPoints)/2
	shift = len(traindata)/2


	#plot for fixed mu=0 training
	print "training fixed"
	clf = svm.NuSVR()
	reducedtrain = 	np.concatenate((traindata[4*chunk : 5*chunk,0], 
		traindata[4*chunk+shift : 5*chunk+shift , 0]))
	reducedtarget = np.concatenate((targetdata[4*chunk : 5*chunk], 
		targetdata[4*chunk+shift : 5*chunk+shift]))

	clf.fit(reducedtrain.reshape((len(reducedtrain),1)), reducedtarget)  
	outputs=clf.predict(traindata[:,0].reshape((len(traindata),1)))
	joblib.dump(clf, 'fixed.pkl') 

	fixedhists=[]
	c1 = ROOT.TCanvas()
	for i, mass in enumerate(massPoints):
		#bkg part
		#plt.hist(outputs[i*chunk+shift: \
		#	(i+1)*chunk+shift], 30, alpha=0.3)
		#sig part
		plt.hist(outputs[i*chunk: \
			(i+1)*chunk], 30, alpha=0.1, range=(-.2,1.2))

		hist = ROOT.TH1F('hist{0}'.format(i),"hist",30,-0.1,1.2)
		fixedhists.append(hist)
		for val in outputs[i*chunk: (i+1)*chunk]:
			hist.Fill(val)
		if i==0:
			hist.Draw()
		else:
			hist.Draw('same')
	c1.SaveAs('roothists.pdf')


	plt.savefig('fixed_training.pdf')
	#plt.show()


	# plot for adaptive training
	print "training adaptive"
	#clf = joblib.load('adaptive.pkl') 
	clf.fit(traindata,targetdata)  
	joblib.dump(clf, 'adaptive.pkl') 

	outputs=clf.predict(traindata)

	#f = ROOT.TFile('workspace_GausSigOnExpBkg.root','r')
	#w = f.Get('w')
	w = ROOT.RooWorkspace('w')
	w.factory('mu[-3,3]')
	bins=30
	low=0.
	high=1.
	w.factory('score[{0},{1}]'.format(low,high))
	s = w.var('score')
	mu = w.var('mu')

	adaptivehists=[]
	adpativedatahists=[]
	adpativehistpdfs=[]
	for i, mass in enumerate(massPoints):
		#bkg part
		#plt.hist(outputs[i*chunk+shift: \
		#	(i+1)*chunk+shift], 30, alpha=0.3)
		#sig part
		plt.hist(outputs[i*chunk: \
			(i+1)*chunk], bins, alpha=0.1, range=(low,high))


		hist = ROOT.TH1F('hist{0}'.format(i),"hist",bins,low,high)
		adaptivehists.append(hist)
		for val in outputs[i*chunk: (i+1)*chunk]:
			hist.Fill(val)
		if i==0:
			hist.Draw()
		else:
			hist.Draw('same')

		datahist = ROOT.RooDataHist('datahist{0}'.format(i),"hist", ROOT.RooArgList(s), hist)
		order=0
		s.setBins(bins)
		histpdf = ROOT.RooHistPdf('histpdf{0}'.format(i),"hist", ROOT.RooArgSet(s), datahist,order)
		histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')

		getattr(w,'import')(histpdf) # work around for morph = w.import(morph)
		getattr(w,'import')(datahist) # work around for morph = w.import(morph)
		adpativedatahists.append(datahist)
		adpativehistpdfs.append(histpdf)

	c1.SaveAs('root_adaptive_hists.pdf')

	w = makeBSpline(w,mu,s,adpativehistpdfs, massPoints)
	w.Print()
	w.writeToFile("workspace_GausSigOnExpBkg_morph.root")
	morph = w.pdf('morph')
	print morph
	morph.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')

	c1 = ROOT.TCanvas("c2",'',400,400)
	frame = s.frame()
	for pdf in adpativehistpdfs:
		pdf.plotOn(frame)
	mu.setVal(0)
	morph.plotOn(frame,ROOT.RooFit.LineColor(ROOT.kRed))
	#morph.plotOn(frame,ROOT.RooFit.LineColor(ROOT.kRed))
	frame.Draw()
	c1.SaveAs('root_bspline.pdf')


	plt.savefig('adaptive_training.pdf')
	#plt.show()
	print massPoints


def testPickl():
	# making plots for when mu is and isn't included in training
	# using data with several mu values
	# FIX: use different test/train samples (not a big deal for simple problem)

	print "Entering testPickl"
	trainAndTarget = np.loadtxt('traindata.dat')
	#to use nmax, need to shuffle first
	traindata = trainAndTarget[:,0:2]
	targetdata = trainAndTarget[:,2]

	massPoints = np.unique(traindata[:,1])
	chunk = len(traindata)/len(massPoints)/2
	shift = len(traindata)/2


	#plot for fixed mu=0 training
	print "training fixed"
	#clf = svm.NuSVR()
	reducedtrain = 	np.concatenate((traindata[4*chunk : 5*chunk,0], 
		traindata[4*chunk+shift : 5*chunk+shift , 0]))
	reducedtarget = np.concatenate((targetdata[4*chunk : 5*chunk], 
		targetdata[4*chunk+shift : 5*chunk+shift]))

	#clf.fit(reducedtrain.reshape((len(reducedtrain),1)), reducedtarget)  
	clf = joblib.load('fixed.pkl') 
	outputs=clf.predict(traindata[:,0].reshape((len(traindata),1)))

	fixedhists=[]
	c1 = ROOT.TCanvas()
	for i, mass in enumerate(massPoints):
		#bkg part
		#plt.hist(outputs[i*chunk+shift: \
		#	(i+1)*chunk+shift], 30, alpha=0.3)
		#sig part
		plt.hist(outputs[i*chunk: \
			(i+1)*chunk], 30, alpha=0.1, range=(-.2,1.2))

		hist = ROOT.TH1F('hist{0}'.format(i),"hist",30,-0.1,1.2)
		fixedhists.append(hist)
		for val in outputs[i*chunk: (i+1)*chunk]:
			hist.Fill(val)
		if i==0:
			hist.Draw()
		else:
			hist.Draw('same')
	c1.SaveAs('roothists.pdf')


	plt.savefig('fixed_training.pdf')
	#plt.show()


	# plot for adaptive training
	print "training adaptive"
	clf = joblib.load('adaptive.pkl') 
	#clf.fit(traindata,targetdata)  
	#joblib.dump(clf, 'adaptive.pkl') 

	outputs=clf.predict(traindata)

	#f = ROOT.TFile('workspace_GausSigOnExpBkg.root','r')
	#w = f.Get('w')
	w = ROOT.RooWorkspace('w')
	w.factory('mu[-3,3]')
	bins=30
	low=0.
	high=1.
	w.factory('score[{0},{1}]'.format(low,high))
	s = w.var('score')
	mu = w.var('mu')

	adaptivehists=[]
	adpativedatahists=[]
	adpativehistpdfs=[]
	for i, mass in enumerate(massPoints):
		#bkg part
		#plt.hist(outputs[i*chunk+shift: \
		#	(i+1)*chunk+shift], 30, alpha=0.3)
		#sig part
		plt.hist(outputs[i*chunk: \
			(i+1)*chunk], bins, alpha=0.1, range=(low,high))


		hist = ROOT.TH1F('hist{0}'.format(i),"hist",bins,low,high)
		adaptivehists.append(hist)
		for val in outputs[i*chunk: (i+1)*chunk]:
			hist.Fill(val)
		if i==0:
			hist.Draw()
		else:
			hist.Draw('same')

		datahist = ROOT.RooDataHist('datahist{0}'.format(i),"hist", ROOT.RooArgList(s), hist)
		order=0
		s.setBins(bins)
		histpdf = ROOT.RooHistPdf('histpdf{0}'.format(i),"hist", ROOT.RooArgSet(s), datahist,order)
		histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')

		getattr(w,'import')(histpdf) # work around for morph = w.import(morph)
		getattr(w,'import')(datahist) # work around for morph = w.import(morph)
		adpativedatahists.append(datahist)
		adpativehistpdfs.append(histpdf)

	c1.SaveAs('root_adaptive_hists.pdf')

	w = makeBSpline(w,mu,s,adpativehistpdfs, massPoints)
	w.Print()
	w.writeToFile("workspace_GausSigOnExpBkg_morph.root")
	morph = w.pdf('morph')
	print morph
	morph.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')

	c1 = ROOT.TCanvas("c2",'',400,400)
	frame = s.frame()
	for pdf in adpativehistpdfs:
		pdf.plotOn(frame)
	mu.setVal(0)
	morph.plotOn(frame,ROOT.RooFit.LineColor(ROOT.kRed))
	#morph.plotOn(frame,ROOT.RooFit.LineColor(ROOT.kRed))
	frame.Draw()
	c1.SaveAs('root_bspline.pdf')


	plt.savefig('adaptive_training.pdf')
	#plt.show()
	print massPoints

def makePdfPlot():
	f = ROOT.TFile('workspace_GausSigOnExpBkg.root','r')
	w = f.Get('w')
	x = w.var('x')
	mu = w.var('mu')
	sigpdf = w.pdf('g')
	bkgpdf = w.pdf('e')
	pdf = w.pdf('model')
	frame = x.frame()
	pdf.plotOn(frame)
	pdf.plotOn(frame,ROOT.RooFit.Components('g'),ROOT.RooFit.LineColor(ROOT.kRed))
	pdf.plotOn(frame,ROOT.RooFit.Components('e'),ROOT.RooFit.LineColor(ROOT.kGreen))
	c1 = ROOT.TCanvas()
	frame.Draw()
	c1.SaveAs('model.pdf')
	f.Close()


def makeData():
	musteps=10
	numTrain=500
	numTest=numTrain

	#Make statistical model
	w = ROOT.RooWorkspace('w')
	w.factory('Gaussian::g(x[-5,5],mu[0,-3,3],sigma[0.5, 0, 2])')
	w.factory('Exponential::e(x,tau[-.15,-3,0])')
	w.factory('SUM::model(s[50,0,100]*g,b[100,0,1000]*e)')
	w.Print() #this isn't displaying in iPython
	w.writeToFile('workspace_GausSigOnExpBkg.root')

	x = w.var('x')
	mu = w.var('mu')
	pdf = w.pdf('model')
	sigpdf = w.pdf('g')
	bkgpdf = w.pdf('e')

	#create training, testing data
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
			testdata1[i+mustep*numTest,1] = 1 # optionally 2*(i%2)-1.

	#print train and test data to file
	np.savetxt('traindata.dat',np.column_stack((traindata,targetdata)), fmt='%f')
	np.savetxt('testdata.dat',testdata, fmt='%f')
	np.savetxt('testdata1.dat',testdata1, fmt='%f')



def KDE():
	# see http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html
	pass

if __name__ == '__main__':
	#makeData()
	#makePdfPlot()
	#trainAndTest()
	#makePdf()
	testPickl()

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
