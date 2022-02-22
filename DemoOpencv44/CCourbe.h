#pragma once
#include<vector>
using namespace std;


// boîte de dialogue de CCourbe

class CCourbe : public CDialogEx
{
	DECLARE_DYNAMIC(CCourbe)

public:
	CCourbe(CWnd* pParent = nullptr);   // constructeur standard
	virtual ~CCourbe();
	int Dessine(vector<double>& x, vector<double>& y);

	

// Données de boîte de dialogue
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG_COURBE };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // Prise en charge de DDX/DDV

	DECLARE_MESSAGE_MAP()
};
