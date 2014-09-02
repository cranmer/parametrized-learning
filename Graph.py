#!/usr/bin/env python
# https://github.com/svenkreiss/PyROOTUtils/blob/master/PyROOTUtils/Graph.py  
__author__ = "Sven Kreiss"
  
import ROOT
from array import array


class Graph( ROOT.TGraph ):
   def __init__( self, x, y=None, fillColor=None, lineColor=None, lineStyle=None, lineWidth=None, markerSize=None, markerColor=None, markerStyle=None, sort=True, name=None, title=None, nameTitle=None ):
      """ takes inputs of the form:
             x = [ (x1,y1), (x2,y2), ... ]
             y = None (default)
          or
             x = [x1,x2,...]
             y = [y1,y2,...]
      """

      if x == None:
         print( "WARNING: Tried to make graph of NULL object. Abort." )
         return

      if isinstance( x,ROOT.TObject ):
         ROOT.TGraph.__init__( self, x )
      
      else:
         if not y:
            # assume x is of the form: [ (x1,y1), (x2,y2), ... ]
            # --> split into xy
            y = [i[1] for i in x]
            x = [i[0] for i in x]
      
         if len(x) != len(y):
            print( "x and y have to have the same length." )
            return
            
         # sort
         if sort:
            xy = sorted( zip(x,y) )
            x = [i for i,j in xy]
            y = [j for i,j in xy]
            
         if len(x) < 1:
            print( "WARNING: trying to create a TGraph without points." )
   
         ROOT.TGraph.__init__( self, len(x), array('d',x), array('d',y) )
         
      if nameTitle: self.SetNameTitle( nameTitle, nameTitle )
      if name: self.SetName( name )
      if title: self.SetTitle( title )
      
      if fillColor:
         self.SetFillColor( fillColor )
      if lineColor:
         self.SetLineColor( lineColor )
      if lineStyle:
         self.SetLineStyle( lineStyle )
      if lineWidth:
         self.SetLineWidth( lineWidth )
      if markerColor:
         self.SetMarkerColor( markerColor )
      if markerStyle:
         self.SetMarkerStyle( markerStyle )
      if markerSize:
         self.SetMarkerSize( markerSize )
         
   def GetRanges( self ):
      r = ( ROOT.Double(), ROOT.Double(), ROOT.Double(), ROOT.Double() )
      self.ComputeRange( r[0], r[1], r[2], r[3] )
      return r

   def eval( self, x ):
      import math
      xyBefore = (-10000,0)
      xyAfter = (10000,0)
      for i in range( 0, self.GetN() ):
         p = ( ROOT.Double(), ROOT.Double() )
         self.GetPoint( i, p[0], p[1] )
         if x < p[0]  and  math.fabs(x-xyBefore[0]) > math.fabs(x-p[0]):
            xyBefore = (p[0],p[1])
         if x > p[0]  and  math.fabs(x-xyAfter[0]) > math.fabs(x-p[0]):
            xyAfter = (p[0],p[1])

      return xyBefore[1]  +   (xyAfter[1]-xyBefore[1])/(xyAfter[0]-xyBefore[0])   *(x-xyBefore[0])

   def transformX( self, function ):
      for i in range( 0, self.GetN() ):
         p = ( ROOT.Double(), ROOT.Double() )
         self.GetPoint( i, p[0], p[1] )
         self.SetPoint( i, function(p[0]), p[1] )

   def transformY( self, function ):
      for i in range( 0, self.GetN() ):
         p = ( ROOT.Double(), ROOT.Double() )
         self.GetPoint( i, p[0], p[1] )
         self.SetPoint( i, p[0], function(p[1]) )

   def derivativeData( self ):
      """ Returns x,y values such that in can be used in Graph constructor. """
      values = []
      for i in range( 1, self.GetN() ):
         p1 = ( ROOT.Double(), ROOT.Double() )
         self.GetPoint( i-1, p1[0], p1[1] )
         p2 = ( ROOT.Double(), ROOT.Double() )
         self.GetPoint( i,   p2[0], p2[1] )

         values.append(  ((p1[0]+p2[0])/2.0, (p2[1]-p1[1])/abs(p2[0]-p1[0]) ) )
      return values

   def derivative2Data( self ):
      """ Returns x,y values such that in can be used in Graph constructor. """
      values = []
      for i in range( 1, self.GetN()-1 ):
         p1 = ( ROOT.Double(), ROOT.Double() )
         self.GetPoint( i-1, p1[0], p1[1] )
         p2 = ( ROOT.Double(), ROOT.Double() )
         self.GetPoint( i,   p2[0], p2[1] )
         p3 = ( ROOT.Double(), ROOT.Double() )
         self.GetPoint( i+1, p3[0], p3[1] )

         point = ( (p1[0]+p3[0])/2.0, (p3[1]-2*p2[1]+p1[1])/(abs(p3[0]-p2[0])*abs(p2[0]-p1[0])) )
         values.append(  point  )
      return values

   def scale( self, factor ):
      self.transformY( lambda y: y*factor )

   def add( self, term ):
      self.transformY( lambda y: y+term )
   
   def integral( self, min=None, max=None ):
      """ Calculate integral using trapezoidal rule. """
      integral = 0.0
      for i in range( 1, self.GetN() ):
         previousPoint = ( ROOT.Double(), ROOT.Double() )
         thisPoint = ( ROOT.Double(), ROOT.Double() )
         self.GetPoint( i-1, previousPoint[0], previousPoint[1] )
         self.GetPoint( i, thisPoint[0], thisPoint[1] )
         
         if min!=None  and  thisPoint[0] < min and previousPoint[0] < min: 
            pass
         elif min!=None  and  thisPoint[0] > min and previousPoint[0] < min: 
            valueAtMin = previousPoint[1]  +  (min-previousPoint[0])*(thisPoint[1]-previousPoint[1])/(thisPoint[0]-previousPoint[0])
            integral += (thisPoint[0]-min) * (thisPoint[1]+valueAtMin)/2.0
         elif max!=None  and  thisPoint[0] > max and previousPoint[0] > max:
            pass
         elif max!=None  and  thisPoint[0] > max and previousPoint[0] < max: 
            valueAtMax = previousPoint[1]  +  (max-previousPoint[0])*(thisPoint[1]-previousPoint[1])/(thisPoint[0]-previousPoint[0])
            integral += (max-previousPoint[0]) * (valueAtMax+previousPoint[1])/2.0
         else:
            integral += (thisPoint[0]-previousPoint[0]) * (thisPoint[1]+previousPoint[1])/2.0
      return integral
      
      
   def argminX( self ):
      """ Get the minimum X. """
      min = 1e30
      minX = None
      for i in range( 0, self.GetN() ):
         p = ( ROOT.Double(), ROOT.Double() )
         self.GetPoint( i, p[0], p[1] )
         if p[1] < min:
            min = p[1]
            minX = p[0]
      return minX
      
   def argminY( self ):
      """ Get the minimum Y. """
      min = 1e30
      for i in range( 0, self.GetN() ):
         p = ( ROOT.Double(), ROOT.Double() )
         self.GetPoint( i, p[0], p[1] )
         if p[1] < min: min = p[1]
      return min
      

   def table( self, bandLow=None, bandHigh=None, bandDifference=True ):
      out = ""
      
      for i in range( self.GetN() ):
         out += "%f \t%f" % ( self.GetX()[i], self.GetY()[i] )
         if bandLow:
            bL = bandLow.Eval( self.GetX()[i] )
            if bandDifference: bL -= self.GetY()[i]
            out += " \t"+str( bL )
         if bandHigh:
            bH = bandHigh.Eval( self.GetX()[i] )
            if bandDifference: bH -= self.GetY()[i]
            out += " \t"+str( bH )
         
         out += "\n"
         
      return out
         
   def getFirstIntersectionsWithGraph( self, otherGraph, xVar=None, xCenter=None, xRange=None, steps=1000 ):
      """ xRange must be of the form (min,max) when given """
      if xVar and not xRange: xRange = (xVar.getMin(), xVar.getMax())
      if xVar and not xCenter: xCenter = xVar.getVal()
      if not xRange: xRange = (self.GetRanges()[0], self.GetRanges()[2])
      
      low,high = (None,None)

      #down
      higher = self.Eval( xCenter ) > otherGraph.Eval( xCenter )
      for i in range( steps+1 ):
         #x = xRange[0]  +   float(i)*( xRange[1]-xRange[0] ) / steps
         x = xCenter  -   float(i)*( xCenter-xRange[0] ) / steps
         
         newHigher = self.Eval( x ) > otherGraph.Eval( x )
         if higher != newHigher:
            low = x
            break
         higher = newHigher
      #up
      higher = self.Eval( xCenter ) > otherGraph.Eval( xCenter )
      for i in range( steps+1 ):
         #x = xRange[0]  +   float(i)*( xRange[1]-xRange[0] ) / steps
         x = xCenter  +   float(i)*( xRange[1]-xCenter ) / steps
         
         newHigher = self.Eval( x ) > otherGraph.Eval( x )
         if higher != newHigher:
            high = x
            break
         higher = newHigher
         
      return (low,high)
      
   def getFirstIntersectionsWithValue( self, value, xVar=None, xCenter=None, xRange=None, steps=1000 ):
      """ xRange must be of the form (min,max) when given """
      if xVar and not xRange: xRange = (xVar.getMin(), xVar.getMax())
      if xVar and not xCenter: xCenter = xVar.getVal()
      if not xRange: xRange = (self.GetRanges()[0], self.GetRanges()[2])
      if not xCenter: xCenter = self.argminX()
      
      low,high = (None,None)

      #down
      higher = self.Eval( xCenter ) > value
      for i in range( steps+1 ):
         #x = xRange[0]  +   float(i)*( xRange[1]-xRange[0] ) / steps
         x = xCenter  -   float(i)*( xCenter-xRange[0] ) / steps
         
         newHigher = self.Eval( x ) > value
         if higher != newHigher:
            low = x
            break
         higher = newHigher         
      #up
      higher = self.Eval( xCenter ) > value
      for i in range( steps+1 ):
         #x = xRange[0]  +   float(i)*( xRange[1]-xRange[0] ) / steps
         x = xCenter  +   float(i)*( xRange[1]-xCenter ) / steps
         
         newHigher = self.Eval( x ) > value
         if higher != newHigher:
            high = x
            break
         higher = newHigher
         
      return (low,high)
      
   def getLatexIntervalFromNll( self, minX=None, up=0.5, xRange=None, steps=1000, digits=2 ):
      """ The parameter up is the same as in a Minos scan (0.5 for nll 
      and 68% two sided intervals). """

      if not minX: minX = self.argminX()
            
      mInterval = self.getFirstIntersectionsWithValue( up, xCenter=minX, xRange=xRange, steps=steps )
      if not mInterval[0]: mInterval = (0,mInterval[1])
      if not mInterval[1]: mInterval = (mInterval[0],0)
      fF = "%."+str(digits)+"f"   # float Format
      return ( (fF+"^{+"+fF+"}_{"+fF+"}") % (minX,mInterval[1]-minX,mInterval[0]-minX) )
