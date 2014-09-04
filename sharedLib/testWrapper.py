from ROOT import *
gSystem.Load( 'libSciKitLearnWrapper' )
#gROOT.LoadMacro( 'MyHelpers.h+' )

def SayHi():
   print 'Hi from python!'
   return 3.

#s = CreateMyClass()
x = RooRealVar('x','x',0,1)
s = SciKitLearnWrapper('s','s',x)
s.RegisterCallBack( SayHi );
s.call_eval()
s.getVal()