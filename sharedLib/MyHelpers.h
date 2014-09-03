class RooAbsReal;
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

//#include "Python.h"

RooAbsReal* CreateMyClass();
void RegisterCallBack( PyObject* );