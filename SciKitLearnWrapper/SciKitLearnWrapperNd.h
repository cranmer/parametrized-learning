/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
  * This code was autogenerated by RooClassFactory                            * 
 *****************************************************************************/

#ifndef SCIKITLEARNWRAPPERND
#define SCIKITLEARNWRAPPERND

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"

#ifndef __CINT__
#include "Python.h"
#endif

#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif


class SciKitLearnWrapperNd : public RooAbsReal{
public:
  SciKitLearnWrapperNd(); // need to implement default constructor {} ; 
  SciKitLearnWrapperNd(const char *name, const char *title, RooArgList& _features);
  SciKitLearnWrapperNd(const SciKitLearnWrapperNd& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new SciKitLearnWrapperNd(*this,newname); }
  inline virtual ~SciKitLearnWrapperNd() { }

  void RegisterCallBack( PyObject* pyobject );

  Double_t call_eval(){return evaluate();}
  Double_t call_getVal(){return getVal();}

protected:

  virtual Double_t evaluate() const ;

private:
  RooListProxy features;
  mutable TIterator* _featureIter ;  //! Iterator over paramSet


  PyObject* m_callback;
  ClassDef(SciKitLearnWrapperNd,1); // Your description goes here...
};
 
#endif
