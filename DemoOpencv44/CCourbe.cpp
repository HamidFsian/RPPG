// CCourbe.cpp : fichier d'implémentation
//

#include "pch.h"
#include "DemoOpencv44.h"
#include "CCourbe.h"
#include "afxdialogex.h"

// boîte de dialogue de CCourbe
// 
// 
// a mettre en VM
double imin, imax;
bool minmaxdefined;

IMPLEMENT_DYNAMIC(CCourbe, CDialogEx)

CCourbe::CCourbe(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG_COURBE, pParent)
{
	minmaxdefined = false;
	imin = 0.0;
	imax = 0.0;
}

CCourbe::~CCourbe()
{
}

void CCourbe::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}




int CCourbe::Dessine(vector<double>& x, vector<double>& y)
{
	
	//Invalidate(true);
	CClientDC dc(this); // device context for painting

	CPen StyloNoir(PS_SOLID, 2, RGB(0, 0, 0));
	CPen StyloRouge(PS_SOLID, 1, RGB(255, 0, 0));
	CPen StyloBleu(PS_SOLID, 2, RGB(0, 0, 255));
	CRect MonRect;
	GetClientRect(&MonRect);
	dc.SelectObject(&StyloNoir);

	CBrush brush;
	brush.CreateSolidBrush(RGB(210, 210, 210));
	dc.FillRect(MonRect, &brush);

	int PourcentMax = 50;
	int Hauteur = MonRect.Height();
	int Largeur = MonRect.Width();
	int i;
	int Taille = x.size();
	double maxy = y[0];
	double miny = y[0];
	double xf, yf, hf;
	for (i = 0; i < Taille; i++)
	{
		if (y[i] > maxy) maxy = y[i];
		if (y[i] < miny) miny = y[i];
	}

	dc.SelectObject(&StyloBleu);

	yf = Hauteur * ((y[0] - miny) / (maxy - miny));
	hf = Hauteur - yf;
	dc.MoveTo(0, hf);


	for (i = 1; i < Taille; i++)
	{
		yf = Hauteur * ((y[i] - miny) / (maxy - miny));
		hf = Hauteur - yf;
		xf = (i * Largeur) / Taille;
		dc.LineTo(xf, hf);
	}

	if (minmaxdefined)
	{
		dc.SelectObject(&StyloRouge);
		xf = (imin * Largeur) / Taille;
		dc.MoveTo(xf, 0);
		dc.LineTo(xf, Hauteur);
		xf = (imax * Largeur) / Taille;
		dc.MoveTo(xf, 0);
		dc.LineTo(xf, Hauteur);
	}
	

	ReleaseDC(&dc);
	return 0;
}

BEGIN_MESSAGE_MAP(CCourbe, CDialogEx)
END_MESSAGE_MAP()


// gestionnaires de messages de CCourbe
