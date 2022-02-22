
// DemoOpencv44Dlg.h : fichier d'en-tête
//
#pragma once

#include"CCourbe.h"

// boîte de dialogue de CDemoOpencv44Dlg
class CDemoOpencv44Dlg : public CDialogEx
{
// Construction
public:
	CDemoOpencv44Dlg(CWnd* pParent = nullptr);	// constructeur standard

// Données de boîte de dialogue
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DEMOOPENCV44_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// Prise en charge de DDX/DDV


// Implémentation
protected:
	HICON m_hIcon;

	// Fonctions générées de la table des messages
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButtonOpen();
	afx_msg void OnBnClickedButtonKppv();
	CString Affichage;
	CString Performance;
	

	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedRandomforest();

	afx_msg void OnBnClickedNn();
};
