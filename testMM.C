void testMM(){
	//	#Make statistical model
	w = new RooWorkspace("w");
	w->factory("Exponential::e(x[-5,15],tau[-.15,-3,0])");
	x = w->var("x");
	w->factory("mu[0,10]");
	mu = w->var("mu");
	frame = x->frame();
	RooArgList pdfs;
	paramVec =	TVectorD(5);
	for( i=0; i<5; ++i){
		w->factory(Form("Gaussian::g%d(x,mu%d[%d,-3,5],sigma[0.5, 0, 2])", i, i,i));
		w->factory(Form("SUM::model%d(s[50,0,100]*g%d,b[100,0,1000]*e)",i,i));
		w->Print() ;
		pdf = w->pdf(Form("model%d",i));
		pdf->plotOn(frame);
		pdfs.add(*pdf);
		paramVec[i]=i;
	}
	pdfs.Print();
	mu = w->var("mu");
	RooArgList varlist;
	varlist.add(*x);
	morph = new RooMomentMorph("morph","morph",*mu,varlist,pdfs, paramVec,RooMomentMorph::Linear);

	w->import(*morph);
	morph->Print("v");

	w->Print();

	for (i=0; i<5; ++i){
		mu->setVal(.8+i);
		mu->Print();
		morph->plotOn(frame, RooFit::LineColor(kRed));
	}
	//	c1 = TCanvas();
	frame->Draw();
	c1->SaveAs("test.pdf");

}