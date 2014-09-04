from ROOT import *

import numpy as np
from sklearn import svm
from sklearn.externals import joblib

def simpleTest():
	print "Simple Test"
	gSystem.Load( 'libSciKitLearnWrapper' )	

	def SayHi():
	   print 'Hi from python!'
	   return 3.	

	x = RooRealVar('x','x',0,1)
	s = SciKitLearnWrapper('s','s',x)
	s.RegisterCallBack( SayHi );
	print "\ngetVal"
	s.getVal()


def inOutFunc(x=0.):
	print "inouttest input was", x
	return x

def inOutTest():
	gSystem.Load( 'libSciKitLearnWrapper' )	
	x = RooRealVar('x','x',0,1)
	s = SciKitLearnWrapper('s','s',x)
	s.RegisterCallBack( inOutFunc );
	print "\ngetVal"
	print s.getVal()


def scikitlearnFunc(x=0.):
	#print "scikitlearnTest"
	clf = joblib.load('../adaptive.pkl') 
	#print "inouttest input was", x
	traindata = np.array((x,0.))
	outputs=clf.predict(traindata)
	#print x, outputs
	return outputs[0]

def scikitlearnTest():
	gSystem.Load( 'libSciKitLearnWrapper' )	
	x = RooRealVar('x','x',0.2,-5,5)	
	#w = RooWorkspace('w')
	#w.factory('x[-1,1]')
	#x = w.var('x')
	s = SciKitLearnWrapper('s','s',x)
	s.RegisterCallBack( scikitlearnFunc );
	print "\ngetVal"
	print s.getVal()

	print s
	print "make plot"
	c1 = TCanvas('c1')
	for val in np.linspace(-1,1,10):
		x.setVal(val)
		temp = s.getVal()
		print "x,s = ", x.getVal(), temp, s.call_getVal()


	frame = x.frame()
	s.plotOn(frame)
	frame.Draw()
	c1.SaveAs('scikitlearn-wrapper-plot.pdf')

if __name__ == '__main__':
	#simpleTest()
	#inOutTest()
	scikitlearnTest()