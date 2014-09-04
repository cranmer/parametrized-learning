from ROOT import *
gSystem.Load( 'libSciKitLearnWrapper' )

def SayHi():
   print 'Hi from python!'
   return 3.

x = RooRealVar('x','x',0,1)
s = SciKitLearnWrapper('s','s',x)
s.RegisterCallBack( SayHi );
print "\ngetVal"
s.getVal()