#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

class SciKitLearnWrapper;
SciKitLearnWrapper* CreateMyClass();
void RegisterCallBack( PyObject* );
