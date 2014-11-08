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
	
			hist = ROOT.TH1F('h{0}hist{1}'.format(name,i),"hist",30,-0.1,1.2)
			fixedhists.append(hist)
			for val in outputs[i*chunk+j*shift: (i+1)*chunk+j*shift]:
				hist.Fill(val)
			if i==0:
				hist.Draw()
			else:
				hist.Draw('same')
	c1.SaveAs('roothists.pdf')




def createPdfForAdaptive_tree(tree):
	'''
	Read in learner saved in adaptive.pkl
	Evaluate outputs for several parameter points, using true value for parameter
	Generate histograms for each point.
	create parametrized pdf that interpolates across these pdfs
	'''
        
	bins=30
	low=0.
	high=1.

        # loop over the tree, build histograms of output for target=(0,1), different mx values
        i=0
        var_points = [ [] , [] ]
        adaptivehists = [ [] , [] ]
        while (tree.GetEntry(i)):
                # have we seen this mx before?
                try:
                        # if so, get the index
                        ind = var_points[int(tree.target)].index(tree.mx)
                except ValueError:

                        # if no, add to our list and make a histogram for it, then get the index
                        var_points[int(tree.target)].append(tree.mx)
                        ind = var_points[int(tree.target)].index(tree.mx)

			hist = ROOT.TH1F('h{0}hist{1}'.format(int(tree.target),ind),"hist",bins,low,high)
			adaptivehists[int(tree.target)].append(hist)

#                if (i%1000==0):
#                        print ' entry ', i , ' mx = ', tree.mx, ' target = ', tree.target, ' ind = ',ind,var_points[0],var_points[1]

                # fill the histogram
                adaptivehists[int(tree.target)][ind].Fill(tree.MLP)
                i=i+1

        # sort them by the var_points
        for target in 0,1:
                var_points[target], adaptivehists[target] = zip(*sorted(zip(var_points[target],adaptivehists[target])))

        print var_points
        print adaptivehists

        # build RooWorld stuff
	w = ROOT.RooWorkspace('w')
	w.factory('mu[{0},{1}]'.format( var_points[0][0],var_points[0][len(var_points[0])-1]))
	w.factory('score[{0},{1}]'.format(low,high))
	s = w.var('score')
	mu = w.var('mu')

        adpativedatahists=[[],[]]
        adpativehistpdfs=[[],[]]
        for target in 0,1:
                for ind in range(0,len(var_points[target])):
                        print "Building RooWorld stuff for target",target," index ",ind
                        print "   mx = ", var_points[target][ind], " mean = ", adaptivehists[target][ind].GetMean(), " rms = ", adaptivehists[target][ind].GetRMS()
                        datahist = ROOT.RooDataHist('dh{0}datahist{1}'.format(target,ind),"hist", 
                                                    ROOT.RooArgList(s), adaptivehists[target][ind])
                
			order=1
			s.setBins(bins)
			histpdf = ROOT.RooHistPdf('hp{0}histpdf{1}'.format(target,ind),"hist", 
				ROOT.RooArgSet(s), datahist,order)
			histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')
	
			getattr(w,'import')(datahist) # work around for morph = w.import(morph)
			getattr(w,'import')(histpdf) # work around for morph = w.import(morph)
			adpativedatahists[target].append(datahist)
			adpativehistpdfs[target].append(histpdf)
		w = makeBSpline(w,mu,s,adpativehistpdfs[target], var_points[target], 'm{0}morph'.format(target))
		morph = w.pdf('m{0}morph'.format(target))
		morph.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')
		print morph

	# make dataset, add to workspace
	w.factory('mwwbb[500,7000]')
	w.factory('mx[350,1600]')
	w.factory('target[-1,2]')
	w.defineSet('inputvars','mwwbb,mx')
	w.defineSet('treevars','mwwbb,mx,target')
	alldata = ROOT.RooDataSet('alldata','',tree, w.set('treevars'))
	getattr(w,'import')(alldata)

	w.Print()
	w.writeToFile("workspace_adaptive_tree.root")


def createPdfForAdaptive():
        f = ROOT.TFile("ttbar_14tev_jes1_eval.root")
        nt = f.Get("nto")
        createPdfForAdaptive_tree(nt)

def plotScore():
	ROOT.gSystem.Load( 'TMVAWrapper/libTMVAWrapper' )
	ROOT.gROOT.ProcessLine(".L RooBSplineBases.cxx+")
	ROOT.gROOT.ProcessLine(".L RooBSpline.cxx+")
	
	f = ROOT.TFile('workspace_adaptive_tree.root','r')
	w = f.Get('w')
	inputVars = ROOT.RooArgList(w.set('inputvars'))
	inputVars.Print()
	nn = ROOT.TMVAWrapper('nn','nn',inputVars,"TMVARegression_ttbar_14tev_jes1.root_MLP.weights.xml")
	frame = w.var('mwwbb').frame()
	for x in np.linspace(400,1600,20):
		w.var('mx').setVal(x)
		nn.plotOn(frame)
	c1 = ROOT.TCanvas("c2",'',400,400)
	frame.Draw()
	c1.SaveAs('tmva.pdf')


def plotAdaptive():
	'''
	make plots of the output of the parametrized model
	'''
	#import class code should work automatically, but confused by namespace
	ROOT.gROOT.ProcessLine(".L RooBSplineBases.cxx+")
	ROOT.gROOT.ProcessLine(".L RooBSpline.cxx+")

	f = ROOT.TFile('workspace_adaptive_tree.root','r')
	w = f.Get('w')

	c1 = ROOT.TCanvas("c2",'',400,400)
	frame = w.var('score').frame()
	c1.SetLogy();

	for val in np.linspace(400,1500,100):
		w.var('mu').setVal(val)
		w.pdf('m1morph').plotOn(frame,ROOT.RooFit.LineColor(ROOT.kRed))
		w.pdf('m0morph').plotOn(frame,ROOT.RooFit.LineColor(ROOT.kBlue))
	frame.Draw()
	c1.SaveAs('root_bspline.pdf')



def fitAdaptive():
	#ugh, tough b/c fixed data are the features, not the NN output
	ROOT.gSystem.Load( 'TMVAWrapper/libTMVAWrapper' )	
	ROOT.gROOT.ProcessLine(".L RooBSplineBases.cxx+")
	ROOT.gROOT.ProcessLine(".L RooBSpline.cxx+")
	ROOT.gROOT.ProcessLine('.L CompositeFunctionPdf.cxx+')

	f = ROOT.TFile('workspace_adaptive_tree.root','r')
	w = f.Get('w')
	w.Print()

	w.factory('CompositeFunctionPdf::sigtemplate(fm0morphfunc)')
	w.factory('CompositeFunctionPdf::bkgtemplate(fm1morphfunc)')
	w.factory('Uniform::baseline(score)')
	w.factory('SUM::template(sigfrac[0,1]*sigtemplate,const[0.01]*baseline,bkgtemplate)')

	mu = w.var('mu')
	mu.setVal(0)

	c1 = ROOT.TCanvas('c1')
	sframe = w.var('score').frame()
	w.pdf('sigtemplate').plotOn(sframe)
	w.pdf('m0morph').plotOn(sframe,ROOT.RooFit.LineColor(ROOT.kGreen))
	w.pdf('m1morph').plotOn(sframe,ROOT.RooFit.LineColor(ROOT.kRed))
	w.pdf('sigtemplate').plotOn(sframe,ROOT.RooFit.LineColor(ROOT.kGreen))
	w.pdf('bkgtemplate').plotOn(sframe,ROOT.RooFit.LineColor(ROOT.kRed))
	w.pdf('template').plotOn(sframe,ROOT.RooFit.LineColor(ROOT.kBlack))
	w.pdf('template').plotOn(sframe,ROOT.RooFit.Components('sigtemplate'),ROOT.RooFit.LineColor(ROOT.kRed))
	w.pdf('template').plotOn(sframe,ROOT.RooFit.Components('bkgtemplate'),ROOT.RooFit.LineColor(ROOT.kGreen))
	sframe.Draw()
	c1.SaveAs('template.pdf')

	#create a dataset for 
	data = w.data('alldata')

	#need a RooAbsReal to evaluate NN(x,mu)
	#nn = ROOT.TMVAWrapper('nn','nn',x,mu)
	print "make RooArgSet"
	print "create TMVAWrapper "
	inputVars = ROOT.RooArgList(w.set('inputvars'))
	inputVars.Print()
	nn = ROOT.TMVAWrapper('nn','nn',inputVars,"TMVARegression_ttbar_14tev_jes1.root_MLP.weights.xml")
	#weightfile = "TMVARegression_alphavary.root_MLP.weights.xml"

	print "about to import"
	print "get val = ",	nn.getVal()
	getattr(w,'import')(ROOT.RooArgSet(nn),ROOT.RooFit.RecycleConflictNodes()) 
	w.Print()	
	print "ok, almost done"
	#create nll based on pdf(NN(x,mu) | mu)
	w.factory('EDIT::pdftemp(template,score=nn)')
	return
	w.factory('EDIT::pdf(pdftemp,mu=mx)')
	w.factory('EDIT::pdf(template,score=nn,mu=mx)')
	#wory that DataHist & HistPdf observable not being reset
	pdf = w.pdf('pdf')
	print 'pdf has expected events = ', pdf.expectedEvents(ROOT.RooArgSet(nn))
	w.Print()
	pdf.graphVizTree('pdf2bTMVA.dot')
	#return

	pdf.fitTo(data,ROOT.RooFit.Extended(False))


	#construct likelihood and plot it
	mu = w.var('mu')
	nll = pdf.createNLL(data,ROOT.RooFit.Extended(False))
	#restrict NLL to relevant region in mu
	frame=mu.frame(-.7,.7)
	nll.plotOn(frame, ROOT.RooFit.ShiftToZero())
	frame.SetMinimum(0)
	frame.SetMaximum(10)
	frame.Draw()
	c1.SaveAs('fitAdaptiveTMVA.pdf')
	return
	


def makeBSpline(w,interpParam, observable, pdfList, paramPoints,name='morph',):
	'''
	The helper function to create the parametrized model that interpolates
	across input pdfs 
	'''
	ROOT.gROOT.ProcessLine(".L RooBSplineBases.cxx+")
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
	morphfunc = ROOT.RooStats.HistFactory.RooBSpline( 'f'+name+'func', "morphfunc", pdfs, bspb, ROOT.RooArgSet() )

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
	plotScore()
	createPdfForAdaptive()
	#plotAdaptive()
	fitAdaptive()
