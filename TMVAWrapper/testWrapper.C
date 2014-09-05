testWrapper(){
	gSystem->Load("libTMVAWrapper.so");
	RooWorkspace w("w");
	w.factory("x[-5,5]");
	w.factory("alpha[-5,5]");
	x = w.var("x");
	alpha = w.var("alpha");
	TMVAWrapper wrap("wrap","wrap",*x, *alpha);
	frame = x->frame();
	wrap.plotOn(frame);
	frame->Draw();
}