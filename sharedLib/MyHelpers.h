#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

class RooAbsReal;
RooAbsReal* CreateMyClass();
void RegisterCallBack( PyObject* );
