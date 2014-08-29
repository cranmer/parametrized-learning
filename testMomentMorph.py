import ROOT
import numpy as np

def makeMomentMorph(w,interpParam, observable, pdfList, paramPoints):
	paramVec = ROOT.TVectorD(len(paramPoints))
	for i, p in enumerate(paramPoints):
		paramVec[i]=p #seems silly, but other constructor gave problems

	pdfs = ROOT.RooArgList()
	for pdf in pdfList:
		pdfs.add(pdf)

	setting = ROOT.RooMomentMorph.Linear
	morph = ROOT.RooMomentMorph('morph','morph',interpParam,
		ROOT.RooArgList(observable),pdfs, paramVec,setting)
	getattr(w,'import')(morph) # work around for morph = w.import(morph)
	return w


def testMomentMorph():

	#Going to make a few statistical models we want to interpolate
	#initialize workspace with some common background part
	w = ROOT.RooWorkspace('w')
	w.factory('Exponential::e(x[-5,15],tau[-.15,-3,0])')
	x = w.var('x')

	frame = x.frame()

	#center of Gaussian will move along the parameter points
	mu = w.factory('mu[0,10]') #this is our continuous interpolation parameter
	paramPoints = np.arange(5)
	pdfs=[]

	# Now make the specific Gaussians to add on top of common background
	for i in paramPoints:
		w.factory('Gaussian::g{i}(x,mu{i}[{i},-3,5],sigma[1, 0, 2])'.format(i=i))
		w.factory('SUM::model{i}(s[50,0,100]*g{i},b[100,0,1000]*e)'.format(i=i))
		w.Print() #this isn't displaying in iPython
		pdf = w.pdf('model{i}'.format(i=i))
		pdfs.append(pdf)
		pdf.plotOn(frame)

	w = makeMomentMorph(w,mu,x,pdfs,paramPoints)
	morph = w.pdf('morph')
	morph.Print('v')

	#make plots of interpolated pdf
	for i in np.arange(5):
		mu.setVal(i+.1) #offset from the original point a bit to see morphing
		mu.Print()
		morph.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kRed))

	c1 = ROOT.TCanvas()
	frame.Draw()
	c1.SaveAs('test.pdf')


def testMomentMorph_orig():

	#Going to make a few statistical models we want to interpolate
	#initialize workspace with some common background part
	w = ROOT.RooWorkspace('w')
	w.factory('Exponential::e(x[-5,15],tau[-.15,-3,0])')
	x = w.var('x')

	frame = x.frame()

	#center of Gaussian will move along the parameter points
	mu = w.factory('mu[0,10]') #this is our continuous interpolation parameter
	paramPoints = np.arange(5)
	pdfs = ROOT.RooArgList()
	#paramVec = ROOT.TVectorD(len(paramPoints),paramPoints) #this gives problems, why?
	paramVec = ROOT.TVectorD(len(paramPoints))

	# Now make the specific Gaussians to add on top of common background
	for i in paramPoints:
		w.factory('Gaussian::g{i}(x,mu{i}[{i},-3,5],sigma[1, 0, 2])'.format(i=i))
		w.factory('SUM::model{i}(s[50,0,100]*g{i},b[100,0,1000]*e)'.format(i=i))
		w.Print() #this isn't displaying in iPython
		pdf = w.pdf('model{i}'.format(i=i))
		pdf.plotOn(frame)
		pdfs.add(pdf)
		paramVec[int(i)]=i

	#ok, now construct the MomentMorph, can choose from these settings
	#  { Linear, NonLinear, NonLinearPosFractions, NonLinearLinFractions, SineLinear } ;
	setting = ROOT.RooMomentMorph.Linear
	morph = ROOT.RooMomentMorph('morph','morph',mu,ROOT.RooArgList(x),pdfs, paramVec,setting)
	getattr(w,'import')(morph) # work around for morph = w.import(morph)
	morph.Print('v')

	#make plots of interpolated pdf
	for i in np.arange(5):
		print i, paramVec[1]
		mu.setVal(i+.5) #offset from the original point a bit to see morphing
		mu.Print()
		morph.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kRed))

	c1 = ROOT.TCanvas()
	frame.Draw()
	c1.SaveAs('test.pdf')


if __name__ == '__main__':
	testMomentMorph()