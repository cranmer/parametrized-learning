from ROOT import *
gSystem.Load( 'libSciKitLearnWrapper' )
gROOT.LoadMacro( 'MyHelpers.h+' )

def SayHi():
   print 'Hi from python!'

s = CreateMyClass()
RegisterCallBack( SayHi );
s.evaluate()
