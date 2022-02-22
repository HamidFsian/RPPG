
// DemoOpencv44.h : fichier d'en-tête principal de l'application PROJECT_NAME
//

#pragma once

#ifndef __AFXWIN_H__
	#error "incluez 'pch.h' avant d'inclure ce fichier pour PCH"
#endif

#include "resource.h"		// symboles principaux


// CDemoOpencv44App :
// Consultez DemoOpencv44.cpp pour l'implémentation de cette classe
//

class CDemoOpencv44App : public CWinApp
{
public:
	CDemoOpencv44App();

// Substitutions
public:
	virtual BOOL InitInstance();

// Implémentation

	DECLARE_MESSAGE_MAP()
};

extern CDemoOpencv44App theApp;
