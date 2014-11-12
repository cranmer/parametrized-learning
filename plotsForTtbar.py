from ROOT import *

def plotsForTtbar():
	f = TFile.Open("ttbar_14tev_alljes_eval.root")
	tree = f.Get('nto')
	c1 = TCanvas("c1","",600,600)
	c1.SetObjectStat(0)
	mx = 1000
	jes = 1
	sameness=""
	colors = [kBlue,kRed]
	hists = []
	for i, target in enumerate([0,1]):
		cuts = 'target=={target} && jes=={jes} && mx=={mx}'.format(jes=jes, target=target,mx=mx)
		print cuts
		histname = 'h{target}{jes}{mx}'.format(jes=jes, target=target,mx=mx)
		hist = TH1F(histname,'',20,0,3000)
		hists.append(hist)
		tree.Project(histname,'mwwbb',cuts)

	for i,hist in enumerate(hists):
		hist.SetLineColor(colors[i])
		hist.SetLineWidth(2)
		hist.SetObjectStat(0)

		#hist.SetFillColor(colors[i])
		hist.DrawNormalized(sameness)
		sameness='same'

	c1.SaveAs('dists_for_ttbar.pdf')


if __name__ == '__main__':
	plotsForTtbar()