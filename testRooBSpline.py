import ROOT
import numpy as np
import Graph

def makeBSpline(w,interpParam, observable, pdfList, paramPoints):
	ROOT.gROOT.ProcessLine(".L RooBSpline.cxx+")

	paramVec = ROOT.TVectorD(len(paramPoints))
	tValues = ROOT.std.vector("double")()
	for i, p in enumerate(paramPoints):
		paramVec[i]=p #seems silly, but other constructor gave problems
		tValues.push_back(p)

	order=3
	bspb = ROOT.RooStats.HistFactory.RooBSplineBases( "bases", "bases", order, tValues, interpParam )


	'''
   	#draw BSplineBases
   	canvas = ROOT.TCanvas('c3')
	axes = canvas.DrawFrame( -3,-0.4,5,1.5 )
	axes.GetXaxis().SetTitle( "m_{H} [GeV]" )
	graphs = []
	for i in range( len(paramPoints) ):
	  	l = ROOT.TLine( paramPoints[i], -0.4, paramPoints[i], -0.1 )
	  	l.SetLineWidth( 4 )
	  	l.SetLineColor( i+1 )
		l.Draw()
	   	graphs.append( l )
	 
		basisPlot = []
		for m in np.linspace(-3,5,20):
			interpParam.setVal(m)
			bspb.getVal()
			print 'xxx', m, bspb.getVal(), bspb.getBasisVal(order,i+(order-order%2)/2,False)
			basisPlot.append( (m, bspb.getBasisVal(order,i+(order-order%2)/2,False)) )
			g = Graph.Graph( basisPlot, lineWidth=2, lineColor=i+1 )
			g.Draw()
			graphs.append( g )

	canvas.SaveAs( "hello.pdf" )
	'''

	pdfs = ROOT.RooArgList()
	for pdf in pdfList:
		pdfs.add(pdf)

	#this makes a function
	morphfunc = ROOT.RooStats.HistFactory.RooBSpline( "morphf", "morphf", pdfs, bspb, ROOT.RooArgSet() )

	#if you want to convert it into a PDF
	rate = w.factory('sum::totalrate(s,b)')
	one = w.factory('one[1.]')
	morph = ROOT.RooRealSumPdf('morph','morph', ROOT.RooArgList(morphfunc), ROOT.RooArgList(one))

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
		pdf.plotOn(frame)

	w.var('tau').setConstant()
	w.var('sigma').setConstant()
	w.var('s').setConstant()
	w.var('b').setConstant()

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
	c1.SaveAs('testBspline1.pdf')

	
	frame2 = x.frame()
	mu.setVal(1.88)
	data = morph.generate(ROOT.RooArgSet(x),10000)
	data = w.pdf('model2').generate(ROOT.RooArgSet(x))
	data.plotOn(frame2)
	morph.fitTo(data)
	morph.plotOn(frame2,ROOT.RooFit.LineColor(ROOT.kRed))
	data = w.pdf('model2').plotOn(frame2)

	mu.setVal(2.)
	print 'morph expectedEvents =', morph.expectedEvents(ROOT.RooArgSet(x))
	print 'model2 expectedEvents =', w.pdf('model2').expectedEvents(ROOT.RooArgSet(x))
	print 'morph get val =', morph.getVal(ROOT.RooArgSet(x))
	print 'model2 get val =', w.pdf('model2').getVal(ROOT.RooArgSet(x))
	frame2.Draw()
	

	c1.SaveAs('testBspline2.pdf')




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