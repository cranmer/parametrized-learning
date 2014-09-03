from ROOT import *

def testConditional():
	w = RooWorkspace("w") 

	gROOT.ProcessLine('.L CompositeFunctionPdf.cxx+')

	# Define (dummy f(x,alpha)
	w.factory('expr::f("x+y+z-3*alpha",x[-10,10],y[-10,10],z[-10,10],alpha[-10,10])')

	# Define a pdf(f) ;
	g = w.factory("Gaussian::g(f,0,1.73)") 
	pdf = w.factory("CompositeFunctionPdf::pdf(g)") 

	# Given a RooDataSet d containing x,alpha
	dummyx = w.factory("Gaussian::dummyx(x,alpha,1)") 
	dummyy = w.factory("Gaussian::dummyy(y,alpha,1)") 
	dummyz = w.factory("Gaussian::dummyz(z,alpha,1)") 
	dummy = w.factory("PROD::dummy(dummyx,dummyy,dummyz)")
	w.defineSet('features','x,y,z')
	w.Print()
	data = dummy.generate(w.set('features'),100)

	# now construct the likelihood
	#RooAbsReal* nll = pdf.createNLL(data,Conditional(w::alpha,kTRUE)) 
	nll = pdf.createNLL(data)
	print "nll = ", nll.getVal()

	c1 = TCanvas('c1')
	frame = w.var('alpha').frame()
	nll.plotOn(frame)
	frame.Draw()
	c1.SaveAs('conditioanl.pdf')
	pdf.fitTo(data)


if __name__ == '__main__':
	testConditional()