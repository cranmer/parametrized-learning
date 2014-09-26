testWrapper(){
	gSystem->Load("libTMVAWrapper.so");
	RooWorkspace w("w");
	w.factory("x[-5,5]");
	w.factory("alpha[-5,5]");
	x = w.var("x");
	alpha = w.var("alpha");
	//	vars = w.allVars();
	RooArgList vars;
	vars.add(*x);
	vars.add(*alpha);
	vars.Print();

	TMVAWrapper wrap("wrap","wrap", vars, "TMVARegression_alphavary.root_MLP.weights.xml");
	wrap.Print("v");
	cout << "wrap = " << wrap.getVal() <<endl;
	x->setVal(1);
	alpha->setVal(2.);
	cout << "wrap = " << wrap.getVal() <<endl;

//	w.import(wrap);
//	w.Print();
//	return;
	//	cout << "\n\nvars are: " << wrap.getListOfVars() << endl;
	frame = x->frame();
	wrap.plotOn(frame);
	frame->Draw();
}