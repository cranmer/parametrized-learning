#!/usr/bin/env python
# https://github.com/svenkreiss/PyROOTUtils/blob/master/PyROOTUtils/Graph.py  
__author__ = "Kyle Cranmer <kyle.cranmer@nyu.edu"
__version__ = "0.1"

'''
This is a research work in progress.

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

import os.path


def makeData():
	'''
	make a RooFit model for some features parametrized by location of Gaussian.
	Use model to create some dummy training and testing data.
	'''
	musteps=10
	numTrain=5000
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

def makeModelPdfPlot():
	'''
	Just make some plots for the RooFit model.
	'''
	print 'Entering makeModelPdfPlot'
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
	c1 = ROOT.TCanvas('c1')
	frame.Draw()
	c1.SaveAs('modelPdfPlots.pdf')
	f.Close()


def trainFixed():
	'''
	train a machine learner based on data from some fixed parameter point.
	save to fixed.pkl
	'''
	print "Entering train fixed"
	trainAndTarget = np.loadtxt('traindata.dat')
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
	joblib.dump(clf, 'fixed.pkl') 

def trainAdaptive():
	'''
	train a machine learner on parametrized data examples.
	save to adaptive.pkl
	'''
	print "Entering train adaptive"
	trainAndTarget = np.loadtxt('traindata.dat')
	traindata = trainAndTarget[:,0:2]
	targetdata = trainAndTarget[:,2]

	massPoints = np.unique(traindata[:,1])
	chunk = len(traindata)/len(massPoints)/2
	shift = len(traindata)/2


	print "training adaptive"
	clf = svm.NuSVR()
	clf.fit(traindata,targetdata)  
	joblib.dump(clf, 'adaptive.pkl') 

def createdPdfForFixed():
	'''
	Read in learner saved in fixed.pkl
	Evaluate outputs for several parameter points.
	Generate histograms for each point.
	(to do:	create parametrized pdf that interpolates across these pdfs)
	'''
	clf = joblib.load('fixed.pkl') 

	trainAndTarget = np.loadtxt('traindata.dat')
	traindata = trainAndTarget[:,0:2]
	targetdata = trainAndTarget[:,2]

	massPoints = np.unique(traindata[:,1])


	fixedhists=[]
	c1 = ROOT.TCanvas()
	for j, name in enumerate(['sig','bkg']):
		for i, mass in enumerate(massPoints):
			#bkg part
			#plt.hist(outputs[i*chunk+shift: \
			#	(i+1)*chunk+shift], 30, alpha=0.3)
			#sig part
	
			hist = ROOT.TH1F('{0}hist{1}'.format(name,i),"hist",30,-0.1,1.2)
			fixedhists.append(hist)
			for val in outputs[i*chunk+j*shift: (i+1)*chunk+j*shift]:
				hist.Fill(val)
			if i==0:
				hist.Draw()
			else:
				hist.Draw('same')
	c1.SaveAs('roothists.pdf')


def createPdfForAdaptive():
	'''
	Read in learner saved in adaptive.pkl
	Evaluate outputs for several parameter points, using true value for parameter
	Generate histograms for each point.
	create parametrized pdf that interpolates across these pdfs
	'''

	trainAndTarget = np.loadtxt('traindata.dat')
	traindata = trainAndTarget[:,0:2]
	targetdata = trainAndTarget[:,2]

	massPoints = np.unique(traindata[:,1])
	chunk = len(traindata)/len(massPoints)/2
	shift = len(traindata)/2

	clf = joblib.load('adaptive.pkl') 

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

	c1 = ROOT.TCanvas("c2",'',400,400)

	for j, name in enumerate(['sig','bkg']):
		adaptivehists=[]
		adpativedatahists=[]
		adpativehistpdfs=[]
		for i, mass in enumerate(massPoints):
			plt.hist(outputs[i*chunk: \
				(i+1)*chunk], bins, alpha=0.1, range=(low,high))
	
	
			hist = ROOT.TH1F('{0}hist{1}'.format(name,i),"hist",bins,low,high)
			adaptivehists.append(hist)
			for val in outputs[i*chunk+j*shift: (i+1)*chunk+j*shift]:
				hist.Fill(val)
			if i==0:
				hist.Draw()
			else:
				hist.Draw('same')
	
			datahist = ROOT.RooDataHist('{0}datahist{1}'.format(name,i),"hist", 
				ROOT.RooArgList(s), hist)
			order=1
			s.setBins(bins)
			histpdf = ROOT.RooHistPdf('{0}histpdf{1}'.format(name,i),"hist", 
				ROOT.RooArgSet(s), datahist,order)
			histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')
	
			getattr(w,'import')(datahist) # work around for morph = w.import(morph)
			getattr(w,'import')(histpdf) # work around for morph = w.import(morph)
			adpativedatahists.append(datahist)
			adpativehistpdfs.append(histpdf)
		w = makeBSpline(w,mu,s,adpativehistpdfs, massPoints, '{0}morph'.format(name))
		morph = w.pdf('{0}morph'.format(name))
		morph.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')
		print morph

		c1.SaveAs('root_adaptive_{0}_hists.pdf'.format(name))

	w.Print()
	w.writeToFile("workspace_adaptive.root")


def plotAdaptive():
	'''
	make plots of the output of the parametrized model
	'''
	#import class code should work automatically, but confused by namespace
	ROOT.gROOT.ProcessLine(".L RooBSpline.cxx+")

	f = ROOT.TFile('workspace_adaptive.root','r')
	w = f.Get('w')
	#w = ROOT.RooWorkspace('w')

	c1 = ROOT.TCanvas("c2",'',400,400)
	frame = w.var('score').frame()
	c1.SetLogy();

	for val in np.linspace(-1,1,20):
		w.var('mu').setVal(val)
		w.pdf('sigmorph').plotOn(frame,ROOT.RooFit.LineColor(ROOT.kRed))
		w.pdf('bkgmorph').plotOn(frame,ROOT.RooFit.LineColor(ROOT.kBlue))
	frame.Draw()
	c1.SaveAs('root_bspline.pdf')


def scikitlearnFunc(x=0.,alpha=0):
	#print "scikitlearnTest"
	clf = joblib.load('adaptive.pkl') 
	#print "inouttest input was", x
	traindata = np.array((x,alpha))
	outputs=clf.predict(traindata)
	#print x, outputs
	#print 'x,alpha =', x, alpha, outputs[0]
	if outputs[0]>1: 
		return 1.
	return outputs[0]

def testSciKitLearnWrapper():
	#need a RooAbsReal to evaluate NN(x,mu)
	ROOT.gSystem.Load( 'SciKitLearnWrapper/libSciKitLearnWrapper' )	
	x = ROOT.RooRealVar('x','x',0.2,-5,5)	
	nn = ROOT.SciKitLearnWrapper('nn','nn',x)
	nn.RegisterCallBack( scikitlearnFunc );

	c1 = ROOT.TCanvas('c1')
	frame = x.frame()
	nn.plotOn(frame)
	frame.Draw()
	c1.SaveAs('testWrapper.pdf')

def testSciKitLearnWrapper2d():
	print "testSciKitLearnWrapper2d"
	#need a RooAbsReal to evaluate NN(x,mu)
	ROOT.gSystem.Load( 'SciKitLearnWrapper/libSciKitLearnWrapper' )	
	x = ROOT.RooRealVar('x','x',0.2,-5,5)	
	mu = ROOT.RooRealVar('mu','mu',0.2,-1,1)	
	#nn = ROOT.SciKitLearnWrapper2d('nn','nn',x,mu)

	nn = ROOT.SciKitLearnWrapperNd('nn','nn',ROOT.RooArgList(x,mu))

	nn.RegisterCallBack( scikitlearnFunc );
	print "callback "
	print "callback ", nn.getVal()
	c1 = ROOT.TCanvas('c1')
	#hist = nn.createHistogram('x,mu',20,20)
	#hist = nn.createHistogram('xmu',x)

	#hist = nn.createHistogram('xmu',x,ROOT.RooFit.Binning(20),
	#	ROOT.RooFit.YVar(mu, ROOT.RooFit.Binning(20)))

	# see http://root.cern.ch/phpBB3/viewtopic.php?t=9643
	#hist = ROOT.RooAbsReal.createHistogram(nn,'x,mu',20,20)
	#hist = ROOT.RooAbsReal.createHistogram(nn,'xmu',x,ROOT.RooFit.Binning(20),
	#	ROOT.RooFit.YVar(mu, ROOT.RooFit.Binning(20)))

	hist = ROOT.TH2F('hist','hist',20,-5,5,20,-1,1)
	for xval in np.linspace(-5,5,20):
		for muval in np.linspace(-1,1,20):
			x.setVal(xval)
			mu.setVal(muval)
			hist.Fill(xval,muval,nn.getVal())
	hist.Draw('surf')
	#hist.Draw()
	c1.SaveAs('testWrapper2d.pdf')
	return

def fitAdaptive():
	#ugh, tough b/c fixed data are the features, not the NN output
	ROOT.gSystem.Load( 'SciKitLearnWrapper/libSciKitLearnWrapper' )	
	ROOT.gROOT.ProcessLine(".L RooBSpline.cxx+")
	ROOT.gROOT.ProcessLine('.L CompositeFunctionPdf.cxx+')

	f = ROOT.TFile('workspace_adaptive.root','r')
	w = f.Get('w')
	#morphfunc = w.pdf('morphfunc')
	w.factory('CompositeFunctionPdf::sigtemplate(sigmorphfunc)')
	w.factory('CompositeFunctionPdf::bkgtemplate(bkgmorphfunc)')
	w.factory('Uniform::baseline(score)')
	w.factory('SUM::template(sigfrac[0,1]*sigtemplate,const[0.01]*baseline,bkgtemplate)')

	mu = w.var('mu')
	mu.setVal(0)

	c1 = ROOT.TCanvas('c1')
	sframe = w.var('score').frame()
	w.pdf('template').plotOn(sframe)
	w.pdf('template').plotOn(sframe,ROOT.RooFit.Components('sigtemplate'),ROOT.RooFit.LineColor(ROOT.kRed))
	w.pdf('template').plotOn(sframe,ROOT.RooFit.Components('bkgtemplate'),ROOT.RooFit.LineColor(ROOT.kGreen))
	sframe.Draw()
	c1.SaveAs('template.pdf')

	#create a dataset for x, shortcut just repeat model
	w.factory('Gaussian::g(x[-5,5],mu,sigma[0.5, 0, 2])')
	w.factory('Exponential::e(x,tau[-.15,-3,0])')
	w.factory('SUM::model(s[50,0,100]*g,b[100,0,1000]*e)')

	x = w.var('x')
	w.Print()
	data = w.pdf('model').generate(ROOT.RooArgSet(x),150)

	#need a RooAbsReal to evaluate NN(x,mu)
	#nn = ROOT.SciKitLearnWrapper2d('nn','nn',x,mu)
	#w.factory('SciKitLearnWrapper::nn(x)')
	#w.factory('SciKitLearnWrapper2d::nn(x,mu)')
	#nn = w.function('nn')
	nn = ROOT.SciKitLearnWrapperNd('nn','nn',ROOT.RooArgList(x,mu))

	nn.RegisterCallBack( scikitlearnFunc );
	print "get val = ",	nn.getVal()
	getattr(w,'import')(ROOT.RooArgSet(nn),ROOT.RooFit.RecycleConflictNodes()) 
	w.Print()

	'''
	#test edit of template (working)
	#wory that DataHist & HistPdf observable not being reset

	w.factory('score2[0,1]')
	w.factory('EDIT::template2(template,score=score2)')
	score2=w.var('score2')
	frame = score2.frame()
	pdf = w.pdf('template2')
	pdf.plotOn(frame)
	c1 = ROOT.TCanvas('c1')
	frame.Draw()
	c1.SaveAs('fitAdaptive.pdf')
	'''
	

	#create nll based on pdf(NN(x,mu) | mu)
	w.factory('EDIT::pdf(template,score=nn)')
	#wory that DataHist & HistPdf observable not being reset
	pdf = w.pdf('pdf')
	print 'pdf has expected events = ', pdf.expectedEvents(ROOT.RooArgSet(nn))
	w.Print()
	pdf.graphVizTree('pdf2b.dot')
	#return

	pdf.fitTo(data,ROOT.RooFit.Extended(False))
	return


	#construct likelihood and plot it
	mu = w.var('mu')
	nll = pdf.createNLL(data,ROOT.RooFit.Extended(False))
	#restrict NLL to relevant region in mu
	frame=mu.frame(-.7,.7)
	nll.plotOn(frame, ROOT.RooFit.ShiftToZero())
	frame.SetMinimum(0)
	frame.SetMaximum(10)
	frame.Draw()
	c1.SaveAs('fitAdaptive.pdf')
	return
	

def makeBSpline(w,interpParam, observable, pdfList, paramPoints,name='morph',):
	'''
	The helper function to create the parametrized model that interpolates
	across input pdfs 
	'''
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
	morphfunc = ROOT.RooStats.HistFactory.RooBSpline( name+'func', "morphfunc", pdfs, bspb, ROOT.RooArgSet() )

	#if you want to convert it into a PDF
	morph = ROOT.RooRealSumPdf(name,name, ROOT.RooArgList(morphfunc), ROOT.RooArgList())

	print morph
	#getattr(w,'import')(morph) # work around for morph = w.import(morph)
	getattr(w,'import')(ROOT.RooArgSet(morph),ROOT.RooFit.RecycleConflictNodes()) # work around for morph = w.import(morph)
	w.importClassCode()

	return w



if __name__ == '__main__':
	'''
	The main function that calls the individual steps of the procedure
	'''
	if  True and os.path.isfile('workspace_GausSigOnExpBkg.root'):
		print 'training data and model already created, skipping makeData()'
	else:
		makeData()

	if os.path.isfile('modelPdfPlots.pdf'):
		print 'model pdf plots already created, skipping makeModelPdfPlot()'
	else:
		pass
		makeModelPdfPlot()

	if os.path.isfile('fixed.pkl'):
		print 'fixed machine learner already created, skipping trainFixed()'
	else:
		trainFixed()

	if os.path.isfile('adaptive.pkl'):
		print 'adaptive machine learner already created, skipping trainAdaptive()'
	else:
		trainAdaptive()

	if True and os.path.isfile('workspace_adaptive.root'):
		print 'adaptive workspace already created, skipping createPdfForAdaptive()'
	else:
		createPdfForAdaptive()

	if True and os.path.isfile('root_bspline.pdf'):
		print 'plots for adatpive already created, skipping plotAdaptive()'
	else:
		plotAdaptive()

	if os.path.isfile('testWrapper.pdf'):
		print 'plots for adatpive already created, skipping testWrapper()'
	else:
		testSciKitLearnWrapper()

	if True and os.path.isfile('testWrapper2d.pdf'):
		print 'plots for adatpive already created, skipping testWrapper2d()'
	else:
		testSciKitLearnWrapper2d()

	if False and os.path.isfile('fitAdaptive.pdf'):
		print 'plots for adatpive already created, skipping fitAdaptive()'
	else:
		fitAdaptive()

