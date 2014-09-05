'''
While I was thinking about the question of reparametrizing the observable of a pdf, I came across a pretty fundamental issue inside of RooFit.

Let’s say I have a pdf(x) and I want to reparametrize x=g(y) and think of it as a pdf in y.
Normally when you do that there is a jacobian factor that comes along.
It seems like this is the point for RooFit performing the normalization integral w.r.t. y.
However, it’s not just an issue of the integral, the shape of the pdf changes too.
In the simple example (below) of Uniform(theta) the pdf of cos(theta) should be 1/sqrt(1-x^2).

Currently in RooFit if I make Uniform(theta) and then reparametrize with theta-> acos(costheta)
and then make  plot of the pdf wrt costheta it is still Uniform. So this is not right if you think of it as reparametrizing the observable. Instead one needs to think of the this type of composition on the observable as a change to the original pdf itself, one that multiplies by the corresponding Jacobian.

Interestingly, this is exactly the behavior we want when it comes to reparametrizing the parameters of the likelihood function, b/c the value of the likelihood is invariant to reparametrization. 

It’s probably pretty hard to change this behavior, but it might be worth making it more clear in the documentation, maybe even issue a light warning.
'''

import ROOT
import numpy as np
import matplotlib.pyplot as plt

def reparamObs():
	w =	ROOT.RooWorkspace('w')
	w.factory('expr::theta("acos(costheta)",costheta[-1,1])')
	w.factory('Uniform::reparam({theta})')
	w.Print()
	costheta=w.var('costheta')
	frame=costheta.frame()
	c1 = ROOT.TCanvas('c1')
	w.pdf('reparam').plotOn(frame)
	frame.Draw()
	c1.SaveAs('reparamObs.pdf')	
	#this should look like 1/sqrt(1-theta^2)

def reparamMC():
	N_MC=100000  # number of Monte Carlo Experiments
	nBins = 50 # number of bins for Histograms
	
	data_x, data_y = [],[]  #lists that will hold x and y
	
	# do experiments
	for i in range(N_MC):
	    # generate observation for x
	    x = np.random.uniform(0,2*np.pi) 
	
	    y = np.cos(x) 
	    data_x.append(x)
	    data_y.append(y)
	
	#setup figures
	fig = plt.figure(figsize=(13,5))
	fig_x = fig.add_subplot(1,2,1)
	fig_y = fig.add_subplot(1,2,2)
	
	fig_x.hist(data_x,nBins,normed=True,range=(0.,2*np.pi))
	fig_x.set_xlabel('angle')
	
	fig_y.hist(data_y,nBins,normed=True)
	fig_y.set_xlabel('cos(angle)')
	
	plt.show()
	plt.savefig('AngleVsCosAngle.pdf')



if __name__ == '__main__':
	reparamObs()
	reparamMC()