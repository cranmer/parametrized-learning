import ROOT
import numpy as np

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
	rate = w.factory('sum::totalrate(s,b)')
	#morph = ROOT.RooRealSumPdf('morph','morph', ROOT.RooArgList(morphfunc), ROOT.RooArgList(rate))
	morph = ROOT.RooRealSumPdf('morph','morph', ROOT.RooArgList(morphfunc), ROOT.RooArgList())

	print morph
	getattr(w,'import')(morph) # work around for morph = w.import(morph)
	return w


def testBSpline():

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
		#w.factory('Gaussian::g{i}(x,mu{i}[{i},-3,5],sigma[1, 0, 2])'.format(i=i))
		w.factory('Gaussian::g{i}(x,mu{i}[{i}],sigma[1, 0, 2])'.format(i=i))
		w.factory('SUM::model{i}(s[50,0,100]*g{i},b[100,0,1000]*e)'.format(i=i))
		w.Print() #this isn't displaying in iPython
		pdf = w.pdf('model{i}'.format(i=i))
		pdfs.append(pdf)
		#pdf.plotOn(frame)

	w = makeBSpline(w,mu,x,pdfs,paramPoints)
	morph = w.pdf('morph')
	w.Print()
	morph.Print()

	#make plots of interpolated pdf
	for i in np.linspace(-3.5,4.5,20):
		mu.setVal(i) #offset from the original point a bit to see morphing
		mu.Print()
		morph.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kRed))

	c1 = ROOT.TCanvas()
	frame.Draw()

	'''
	frame2 = x.frame()
	mu.setVal(2.)
	data = morph.generate(ROOT.RooArgSet(x),10000)
	data.plotOn(frame2)
	morph.fitTo(data)
	morph.plotOn(frame2)
	print 'expectedEvents =', morph.expectedEvents(ROOT.RooArgSet(x))
	frame2.Draw()
	'''

	c1.SaveAs('testBspline.pdf')




if __name__ == '__main__':
	testBSpline()


	'''from Sven:
	tValues = ROOT.std.vector("double")()
    for m in masses: tValues.push_back( m )
    	order = 3
	    bspb = ROOT.RooStats.HistFactory.RooBSplineBases( "bases", "bases", order, tValues, w.var("mH") )

	listOfNorms = ROOT.RooArgList( "listOfNorms" )

    for n in normsRRV: listOfNorms.add( n )
      sf_scale_bsNoLimit = ROOT.RooStats.HistFactory.RooBSpline( "sf_scale_bs_%s_noLimit" % t, "sf_scale_bs_%s_noLimit" % t, listOfNorms, bspb, ROOT.RooArgSet() )
    '''