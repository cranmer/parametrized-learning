// @(#)root/roostats:$Id:  armbrust $
// Author: Aaron Armbruster
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ROOBSPLINEBASES
#define ROOSTATS_ROOBSPLINEBASES

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"

#include <sstream>
#include <vector>
using namespace std;

class RooRealVar;
class RooArgList ;

namespace RooStats{
namespace HistFactory{

  class RooBSplineBases : public RooAbsReal {
  public:

    RooBSplineBases() ;
    RooBSplineBases(const char* name, const char* title, int order, vector<double>& tValues,
		    RooAbsReal& t, int nrClose=0);

    RooBSplineBases(const char *name, const char *title);
    RooBSplineBases(const RooBSplineBases&, const char*);

    virtual TObject* clone(const char* newname) const { return new RooBSplineBases(*this, newname); }
    virtual ~RooBSplineBases() ;

/*     Double_t getCurvature() const; */

    int getOrder() const {return _n;}
    //void setWeights(const RooArgList& weights);
    Double_t getBasisVal(int n, int i, bool rebuild=true) const;
    vector<double> getTValues() const {return _tValues;}
    vector<double> getTAry() {return _t_ary;}

  protected:

    void buildTAry() const;

    //RooListProxy _controlPoints;
    vector<double> _tValues;
    //RooListProxy _t_ary;
    int _m;
/*     mutable double* _t_ary; //[_m] */
    RooRealProxy _t;
    int _n;
    int _nrClose;
    //int _nPlusOne;
    //mutable double** _bin; //[_nPlusOne][_m]
    mutable vector<double> _t_ary;
    mutable vector<vector<double> > _bin;

    Double_t evaluate() const;

    ClassDef(RooStats::HistFactory::RooBSplineBases,1) // Uniform B-Spline
  };
}
}

#endif
