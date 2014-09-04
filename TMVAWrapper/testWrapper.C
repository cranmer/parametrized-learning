testWrapper(){
	gSystem->Load("libTMVAWrapper.so");
	RooWorkspace w("w");
	w.factory("x[-5,5]");
	x = w.var("x");
	TMVAWrapper wrap("wrap","wrap",*x);
	frame = x->frame();
	wrap.plotOn(frame);
	frame->Draw();
}