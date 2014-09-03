
#include "SciKitLearnWrapper.h"
#include "Python.h"
#include "RooAbsReal.h"

static SciKitLearnWrapper s_myClass;

RooAbsReal* CreateMyClass() {
   return &s_myClass;
}

void RegisterCallBack( PyObject* pyobject )
{
   s_myClass.RegisterCallBack( pyobject );
}
