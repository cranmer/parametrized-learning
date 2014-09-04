test(){
	gSystem->Load("libSciKitLearnWrapper.so");
	RooWorkspace w("w");
	w.factory("x[-5,5]");
	w.factory("mu[-5,5]");
	w.defineSet("myset","x,mu")
	w.factory("SciKitLearnWrapper::nn(x)");
	RooArgList list;
	list.add(*w.var("x"));
	list.add(*w.var("mu"));
	test = SciKitLearnWrapperNd("test","test",list);
	w.import(test);
	test2 = SciKitLearnWrapper2d("test2","test",*w.var("x"),*w.var("mu"));
	w.import(test2);
	w.Print();

	//	w.factory("SciKitLearnWrapper2d::nndd(x,mu)");

}
