
// DemoOpencv44Dlg.cpp : fichier d'implémentation
//

#include "pch.h"
#include "framework.h"
#include "DemoOpencv44.h"
#include "DemoOpencv44Dlg.h"
#include "afxdialogex.h"
#include "CCourbe.h"
#include <string>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/ml.hpp"

using namespace cv;
using namespace cv::ml;
#ifdef _DEBUG
#define new DEBUG_NEW
#endif
using namespace cv;
using namespace std;
int _brightness = 100;
int _contrast = 100;

#define SSTR( x ) static_cast< std::ostringstream & >( \ ( std::ostringstream() << std::dec << x ) ).str()

struct SPerform
{
	float TP;
	float TN;
	float FP;
	float FN;

	
};

struct SMetric
{
	float Precision;
	float Accuracy;
	float Recall;
	float F_Score;
};
//compteur faux positif faux negatif
SPerform Perf = { 0,0,0,0 };
SMetric Metric = { 0,0,0,0 };



void GenData(Mat& D, Mat& L, int cl, int sample_count, int m1, int m2){
	D.create(sample_count, 2, CV_32FC1);
	L.create(sample_count, 1, CV_32FC1);
	RNG rng(12345);
	rng.fill(D.col(0), RNG::NORMAL, Scalar(m1), Scalar(m2)); // abscisse
	rng.fill(D.col(1), RNG::NORMAL, Scalar(m1), Scalar(m2)); // ordonnee
	for (int i = 0; i < L.rows; ++i) L.at<int>(i) = cl;

}

SMetric CalculMetric(SPerform Perf)
{
	Metric.Accuracy = (Perf.TP + Perf.TN) / (Perf.FN + Perf.FP + Perf.TN + Perf.TP);
	Metric.Precision = (Perf.TP) / (Perf.TP + Perf.FP);
	Metric.Recall = (Perf.TP) / (Perf.TP + Perf.FN);
	Metric.F_Score = (Metric.Accuracy * Metric.Recall) / (Metric.Accuracy + Metric.Recall);
	return Metric;
	
};
	

// Fourier 
bool FFT(int dir, int m, double* x, double* y)
{
	long nn, i, i1, j, k, i2, l, l1, l2;
	double c1, c2, tx, ty, t1, t2, u1, u2, z;

	/* Calculate the number of points */
	nn = 1 << m;

	/* Do the bit reversal */
	i2 = nn >> 1;
	j = 0;
	for (i = 0; i < nn - 1; i++) {
		if (i < j) {
			tx = x[i];
			ty = y[i];
			x[i] = x[j];
			y[i] = y[j];
			x[j] = tx;
			y[j] = ty;
		}
		k = i2;
		while (k <= j) {
			j -= k;
			k >>= 1;
		}
		j += k;
	}

	/* Compute the FFT */
	c1 = -1.0;
	c2 = 0.0;
	l2 = 1;
	for (l = 0; l < m; l++) {
		l1 = l2;
		l2 <<= 1;
		u1 = 1.0;
		u2 = 0.0;
		for (j = 0; j < l1; j++) {
			for (i = j; i < nn; i += l2) {
				i1 = i + l1;
				t1 = u1 * x[i1] - u2 * y[i1];
				t2 = u1 * y[i1] + u2 * x[i1];
				x[i1] = x[i] - t1;
				y[i1] = y[i] - t2;
				x[i] += t1;
				y[i] += t2;
			}
			z = u1 * c1 - u2 * c2;
			u2 = u1 * c2 + u2 * c1;
			u1 = z;
		}
		c2 = sqrt((1.0 - c1) / 2.0);
		if (dir == 1)
			c2 = -c2;
		c1 = sqrt((1.0 + c1) / 2.0);
	}

	/* Scaling for forward transform

	if (dir == 1) {
		for (i=0;i<nn;i++) {
			x[i] /= (double)nn;
			y[i] /= (double)nn;
		}
	}
	*/
	return true;
}

//Max 

int getMaxElement(vector<double>& array, int sizeOfArray) //passer comme argument matrice et taille matrice
{
	//init
	int resultingMaxValue = 0;
	int indx=0;
	//verifier
	if (sizeOfArray > 0) {
		resultingMaxValue = array[0]; //init

		for (int i = 0; i < sizeOfArray; ++i) {
			if (array[i] > resultingMaxValue)
			{
				//mise a jour du resultat et de l'index
				resultingMaxValue = array[i];
				indx = i;
			}
		}
	}
	return indx;
}
/*void mlp(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {
	cv::Mat layers = cv::Mat(4, 1, CV_32SC1);
	layers.row(0) = cv::Scalar(2);
	layers.row(1) = cv::Scalar(10);
	layers.row(2) = cv::Scalar(15);
	layers.row(3) = cv::Scalar(1);

	cv::Ptr<cv::ml::ANN_MLP> mlp;
	CvANN_MLP_TrainParams params;
	CvTermCriteria criteria;
	criteria.max_iter = 100;
	criteria.epsilon = 0.00001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.05f;
	params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;
	mlp.create(layers);
	// train
	mlp.train(trainingData, trainingClasses, cv::Mat(), cv::Mat(), params);
	cv::Mat response(1, 1, CV_32FC1);
	cv::Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		cv::Mat response(1, 1, CV_32FC1);
		cv::Mat sample = testData.row(i);
		mlp.predict(sample, response);
		predicted.at < float >(i, 0) = response.at < float >(0, 0);
	}
	cout << " Accuracy_ { MLP } = " << evaluate(predicted, testClasses) << endl;
	plot_binary(testData, predicted, " Predictions Backpropagation ");
}*/
// boîte de dialogue CAboutDlg utilisée pour la boîte de dialogue 'À propos de' pour votre application

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Données de boîte de dialogue
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // Prise en charge de DDX/DDV

// Implémentation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// boîte de dialogue de CDemoOpencv44Dlg



CDemoOpencv44Dlg::CDemoOpencv44Dlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DEMOOPENCV44_DIALOG, pParent)
	, Affichage(_T(""))
	, Performance(_T(""))
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CDemoOpencv44Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_TEXTE, Affichage);
	DDX_Text(pDX, IDC_TEXT2, Performance);
}

BEGIN_MESSAGE_MAP(CDemoOpencv44Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_OPEN, &CDemoOpencv44Dlg::OnBnClickedButtonOpen)
	ON_BN_CLICKED(IDC_BUTTON_KPPV, &CDemoOpencv44Dlg::OnBnClickedButtonKppv)
	ON_BN_CLICKED(IDC_BUTTON2, &CDemoOpencv44Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_RandomForest, &CDemoOpencv44Dlg::OnBnClickedRandomforest)
	ON_BN_CLICKED(IDC_NN, &CDemoOpencv44Dlg::OnBnClickedNn)
END_MESSAGE_MAP()


// gestionnaires de messages de CDemoOpencv44Dlg

BOOL CDemoOpencv44Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Ajouter l'élément de menu "À propos de..." au menu Système.

	// IDM_ABOUTBOX doit se trouver dans la plage des commandes système.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Définir l'icône de cette boîte de dialogue.  L'infrastructure effectue cela automatiquement
	//  lorsque la fenêtre principale de l'application n'est pas une boîte de dialogue
	SetIcon(m_hIcon, TRUE);			// Définir une grande icône
	SetIcon(m_hIcon, FALSE);		// Définir une petite icône

	// TODO: ajoutez ici une initialisation supplémentaire

	return TRUE;  // retourne TRUE, sauf si vous avez défini le focus sur un contrôle
}

void CDemoOpencv44Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// Si vous ajoutez un bouton Réduire à votre boîte de dialogue, vous devez utiliser le code ci-dessous
//  pour dessiner l'icône.  Pour les applications MFC utilisant le modèle Document/Vue,
//  cela est fait automatiquement par l'infrastructure.

void CDemoOpencv44Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // contexte de périphérique pour la peinture

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Centrer l'icône dans le rectangle client
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Dessiner l'icône
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// Le système appelle cette fonction pour obtenir le curseur à afficher lorsque l'utilisateur fait glisser
//  la fenêtre réduite.
HCURSOR CDemoOpencv44Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CDemoOpencv44Dlg::OnBnClickedButtonOpen()
{
	//créa de boite dialogue secondaire
	CCourbe* pSignal = new CCourbe(this);
	pSignal->Create(IDD_DIALOG_COURBE);
	pSignal->ShowWindow(true);

	Mat frame;
	Mat MaCopie;
	vector<double> MoyB;
	int TailleBuf=32;
	MoyB.resize(TailleBuf);
	

	for (int u = 0; u < TailleBuf; u++)
	{
		MoyB[u] = 0.0; //init de la moyenne
	}
	//"JM_RPPG_P1001511.mp4"
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return;
	//Mat Blur;
	Mat edges;
	Mat edgesFlou;
	Mat Contraste;
	CascadeClassifier face_cascade;
	face_cascade.load("C:\\opencv451\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml"); //upload cascade de Haar
	std::vector<Rect> faces; // vecteur faces de type vecteur
	double tickf = getTickFrequency(); //pour compter
	// vecteur pour FFT
	double* real = (double*)malloc(TailleBuf * sizeof(double));
	double* imag = (double*)malloc(TailleBuf * sizeof(double));//calloc(TailleBuf,0)
	for (;;) //boucle infinie
	{
		double timer = (double)getTickCount();
		cap >> frame; // get a new frame from camera
		//resize(frame, frame, Size(frame.cols / 2, frame.rows / 2));
		Mat MaCopie(frame);
		int Largeur = frame.cols;
		int Hauteur = frame.rows;
		cvtColor(frame, edges, cv::COLOR_BGR2GRAY);

		// face detection
		Mat rgbchannel[3]; //pour séparer les canaux
		Mat rgchannelRegion[3];
		Mat image_blurred;
		Mat image_blurredV2; 
		Mat	MoyTEST;
		Mat FFTres(Size(TailleBuf, 1), CV_64F);

		double freqcar;
		Scalar Moyenne;
		Scalar MoyenneBlur;
		Scalar MoyRes;
		Scalar MoyResv2;

		face_cascade.detectMultiScale(frame, faces, 1.1, 3, 0, Size(20, 20)); //detectoin visage
		if (faces.size()>0) //pour verifier
		{
			rectangle(frame, faces[0], Scalar(255, 255, 255), 1, 1, 0); // dessiner un rectangle
			// The actual splitting.
			split(frame, rgbchannel); //on split les canaux en 3 (rouge vert bleu)
			// aprés avoir séparer les canaux, on réduit la hauteur a 70% et la largeur a 90%
			rgchannelRegion[0] = rgbchannel[0](Rect(Point(faces[0].x + 30, faces[0].y), Point( (faces[0].x + 0.7 * faces[0].width),  (faces[0].y + 0.9 * faces[0].height)))); // Rouge
			rgchannelRegion[1] = rgbchannel[1](Rect(Point(faces[0].x + 30, faces[0].y), Point((faces[0].x + 0.7 * faces[0].width), (faces[0].y + 0.9 * faces[0].height))));// green
			rgchannelRegion[2] = rgbchannel[2](Rect(Point(faces[0].x + 30, faces[0].y), Point((faces[0].x + 0.7 * faces[0].width),  (faces[0].y + 0.9 * faces[0].height))));//Blue
			//calcul de la moyenne
			
			MoyB.erase(MoyB.begin()); //effacer la premiere val
			Moyenne = mean(rgchannelRegion[1]);// faire la moyenne du canal vert
			MoyB.push_back(Moyenne[0]);// faire entrer la nouvelle val de la moyenne à l'emplacement 0
			Mat LowFiter;
			Mat Inter;
			Mat MoyBMat(Size(TailleBuf,1),CV_64F);
			Mat temp(Size(TailleBuf, 1), CV_64F);

			vector<double> FFTvect(TailleBuf); //ça c'est pour voir la tot du FFT 
			vector<double> FFTvectNorm(TailleBuf);// pour voir la FFT filtré
			vector<double> FFTvectNew(TailleBuf); //pour voir que de 0 à 50

			vector<double> data_x(TailleBuf);
			vector<double> data_y(TailleBuf);

			Mat MoyBFiltre;
			Mat MoyBFiltre2; //sortie 

			for (int i = 0; i < TailleBuf; i++)
			{
				temp.at<double>(i) = MoyB[i]; //parce que MoyB est un vecteur et on le change en mat pour pouvoir dessiner
			}

			GaussianBlur(temp, LowFiter, Size(3, 3), 50);
			GaussianBlur(LowFiter, Inter, Size(3, 3), 50);
			GaussianBlur(Inter, LowFiter, Size(3, 3), 50); //on filtre à trois reprise car GaussianBlur n'admets pas de kernel plus de (9,9)
			
			MoyBFiltre = temp - LowFiter; // refiltré le signal
			GaussianBlur(MoyBFiltre, MoyBFiltre2, Size(3,3), 1); //on filtre notre dernier signal avec un param 1 pour eliminer HF

			for (int i = 0; i < TailleBuf; i++)
			{
				//temp.at<double>(i) = MoyBFiltre2(i); //parce que MoyB est un vecteur et on le change en mat pour pouvoir dessiner
				real[i] = MoyBFiltre2.at<double>(i); //mettre le signal filtré finit dans les réels
				imag[i] = 0;
			}

			bool fftres = FFT(1, log2(TailleBuf), real, imag);// calculer la FFT en fonction de la plage, les real et les imaginaires
			//FFTres = fftres; //stockez le rest de FFT dans une Mat    faux
					
			for (int i = 0; i < TailleBuf; i++)
			{ 
				double Module= sqrt(real[i] * real[i] + imag[i]*imag[i]); //calcul du module
				FFTres.at<double>(i) = Module;
				//MoyTEST = MoyenneBlur[i];
				FFTvect[i] = FFTres.at<double>(i); // ou sinon mettre Module 
				data_x[i] = double(i);
				data_y[i] = Module;
			}
			
			for (int i = 0; i < (TailleBuf/2)-13; ++i) //prendre qu'un intervalle
			{
				FFTvectNew[i] = FFTvect[i+1]; //pour éviter le premier pic important
			}
			// Calcul de la freq cardiaque
			double Pe = 0.06;
			double tempsglobal = Pe * TailleBuf; 
			double tempsechminute = tempsglobal / 60.0;
			int TheMax = getMaxElement(FFTvectNew, FFTvectNew.size()); //pour avoir le max à chaque iteration
			freqcar = double(TheMax) / double(tempsechminute);
			//pSignal->Dessine(data_x, data_y); //1er vis
			//pSignal->Dessine(data_x, MoyBFiltre2); // pour voir la moyenne filtré
			//pSignal->Dessine(data_x, FFTvect); // pour voir la fft 
			pSignal->Dessine(data_x, FFTvectNew); // pour voir FFT normaliser
		} // fin si un visage trouvé

		std::string s = "bpm : " + std::to_string(freqcar);
		putText(frame, s, Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 2); //pour afficher le texte dans l'image d'origine
		imshow("FACE", frame); 

		//imshow("Red", rgchannelRegion[0]);
		//imshow("Green", rgchannelRegion[1]);
		//imshow("Blue", rgchannelRegion[2]);

		if (waitKey(0x1B) >= 0) break;
		float temps;
		float Pe = 0.08; //trés important psk pour cadenser les images
		float fps;
		do
		{
			fps = tickf / ((double)getTickCount() - timer); temps = 1.0 / fps;
		} while (temps < Pe); // on attend 80ms, donc fixez Pe = 0.08
	}
	//free memory malloc
	free(real);
	free(imag);
	destroyAllWindows();
	// the camera will be deinitialized automatically in VideoCapture destructor
}


void CDemoOpencv44Dlg::OnBnClickedButtonKppv()
{
	//train
	const int K = 7;
	int li, col, k, accuracy;
	int train_sample_count = 100; //iteration
	Mat trainData1(train_sample_count, 2, CV_32FC1); //cv : type opencv, 32:bits, F: float, C: canal , 1 : nombre de canaux
	Mat trainData2(train_sample_count, 2, CV_32FC1);
	Mat TrainClasses1 = Mat::zeros(train_sample_count, 1, CV_32FC1); //train our classe with a matrice of zeros with size(100,1)
	Mat TrainClasses2 = Mat::ones(train_sample_count, 1, CV_32FC1);//méme chose mais avec matrice de 1
	Mat trainDataGlobal;
	Mat TrainClasseG;
	Mat Img = Mat::zeros(500, 500, CV_8UC3); //Image sera une matrice de 0 (500,500) qui est notre espace
	Mat Echantillon(1, 2, CV_32FC1);
	RNG rng(12345); // permettant de faire un tirage aléatoire
	Mat PlusProche(1, K, CV_32FC1);  //matrice (1,K) de type float sur 32bits et un canal
	Mat Reponse, Dist; //afficher la reponse ? !
	rng.fill(trainData1.col(0), RNG::NORMAL, Scalar(50), Scalar(80)); // abscisse
	rng.fill(trainData1.col(1), RNG::NORMAL, Scalar(150), Scalar(80)); // ordonnee
	rng.fill(trainData2.col(0), RNG::NORMAL, Scalar(300), Scalar(80));// abscisse
	rng.fill(trainData2.col(1), RNG::NORMAL, Scalar(190), Scalar(80));// ordonnee
	vconcat(trainData1, trainData2, trainDataGlobal);
	vconcat(TrainClasses1, TrainClasses2, TrainClasseG);

	


	
	//test
	int test_sample_count = 1000;
	Mat TestData1(test_sample_count, 2, CV_32FC1); //cv : type opencv, 32:bits, F: float, C: canal , 1 : nombre de canaux
	Mat TestData2(test_sample_count, 2, CV_32FC1);
	//ne sert a rien
	 //Mat TestClasses1 = Mat::zeros(test_sample_count, 1, CV_32FC1); //train our classe with a matrice of zeros with size(100,1)
	 //Mat TestClasses2 = Mat::ones(test_sample_count, 1, CV_32FC1);
	Mat TestDataGlobal;
	Mat TestClasseG;
	 // permettant de faire un tirage aléatoire
	


	rng.fill(TestData1.col(0), RNG::NORMAL, Scalar(30), Scalar(50)); // abscisse
	rng.fill(TestData1.col(1), RNG::NORMAL, Scalar(190), Scalar(50)); // ordonnee
	rng.fill(TestData2.col(0), RNG::NORMAL, Scalar(240), Scalar(50));// abscisse
	rng.fill(TestData2.col(1), RNG::NORMAL, Scalar(100), Scalar(50));// ordonnee
	vconcat(TestData1, TestData2, TestDataGlobal);
	


	 //Pour transférer ces données au classifieur :
	  Ptr<ml::KNearest> knn = ml::KNearest::create();
	 knn->train(trainDataGlobal, ml::ROW_SAMPLE, TrainClasseG);
	 //Il faut donc faire varier li et col de 0 à la largeur et hauteur de l’image représentant l’espace pour 
	 //visualiser la frontière (on classe tous les points de l’espace discrétisé).
	 for (int li = 0; li <Img.rows; li++)
	 {
		 for (int col = 0; col < Img.cols; col++)
		 {
			 Echantillon.at<float>(0) = (float)(col);
			 Echantillon.at<float>(1) = (float)(li);
			 //Pour estimer la réponse d’un Kppv en un point de coordonnée (li,col) de l’espace :
			 float rr = knn->findNearest(Echantillon, K, Reponse, PlusProche, Dist);
			 
			 if (rr == 1)
			 {
				 Img.at<Vec3b>(li, col) = Vec3b(180, 0, 0);
			 }
			 else
			 {
				 Img.at<Vec3b>(li, col) = Vec3b(0, 180, 0);
			 }
			 
		 }
	 }
	 //Enfin pour afficher les échantillons d’origine et l’image résultante :
	 for (int i = 0; i < train_sample_count; i++)
	 {
		 Point pt; //création d'un point
		 pt.x = trainData1.at<float>(i, 0); //coordonées X
		 pt.y = trainData1.at<float>(i, 1); // Y

		 circle(Img, pt, 2, CV_RGB(255, 0, 0), FILLED); //dessiner un cerle rouge derayon 2 à partir du pt

		 pt.x = trainData2.at<float>(i, 0);//autre point X
		 pt.y = trainData2.at<float>(i, 1); // Y

		 circle(Img, pt, 2, CV_RGB(0, 255, 0), FILLED); //autre cercle verte
	 }
	 //On donne à chaque point de l’image une couleur correspondant à la classe 
	 //trouvée par le Knn.
	 
	 imshow("res", Img);
	 waitKey(0);

	 

	 for (int i = 0; i < TestData1.rows; i++)
	 {
		 Mat res;
		 Echantillon.at<float>(0) = TestData1.at<float>(i,0);
		 Echantillon.at<float>(1) = TestData1.at<float>(i, 1);
		 // predict on majority of k PlusProche:

		float cl = knn->findNearest(Echantillon, K, Reponse, PlusProche, Dist);
		if (cl == 0)
			Perf.TN++;
		else
			Perf.FP++;
		
	 }
	 for (int i = 0; i < TestData2.rows; i++)
	 {
		 Mat res;
		 Echantillon.at<float>(0) = TestData2.at<float>(i, 0);
		 Echantillon.at<float>(1) = TestData2.at<float>(i, 1);
		 // predict on majority of k PlusProche:

		 float cl = knn->findNearest(Echantillon, K, Reponse, PlusProche, Dist);
		 if (cl == 1)
			 Perf.TP++;
		 else
			 Perf.FN++;

	 }
	

	
	//struct Smetrics//retour var Smetrics
	 

	SMetric Res = CalculMetric(Perf);
		
	Affichage.Format(L" METRICS \n  Accuracy = %f \n Precision= %f \n Recall = %f \n F_Score = %f " ,Metric.Accuracy, Metric.F_Score , Metric.Precision, Metric.Recall );
    Performance.Format(L" \n Performance \n TP=%f \n TN=%f \n FP=%f \n FN=%f  ", Perf.TN, Perf.FN, Perf.TP ,Perf.FP);
		UpdateData(false);
		int kk = 0;
}



// SVM
void CDemoOpencv44Dlg::OnBnClickedButton2()
{
	// the main problem was in the lables we had to put CV_32SC1 it mean signed int
	// INIT
	int li, col, accuracy;
	int train_sample_count = 100; //iteration
	Mat trainData1(train_sample_count, 2, CV_32FC1); //cv : type opencv, 32:bits, F: float, C: canal , 1 : nombre de canaux
	Mat trainData2(train_sample_count, 2, CV_32FC1);
	Mat TrainClasses1 = Mat::zeros(train_sample_count, 1, CV_32SC1); //train our classe with a matrice of zeros with size(100,1)
	Mat TrainClasses2 = Mat::ones(train_sample_count, 1, CV_32SC1);//méme chose mais avec matrice de 1
	Mat trainDataGlobal;
	Mat TrainClasseG;
	Mat ImgSVM = Mat::zeros(500, 500, CV_8UC3); //Image sera une matrice de 0 (500,500) qui est notre espace
	Mat Echantillon(1, 2, CV_32FC1);
	RNG rng(12345); // permettant de faire un tirage aléatoire

	
	Mat Reponse, Dist; //afficher la reponse ? !
	rng.fill(trainData1.col(0), RNG::NORMAL, Scalar(50), Scalar(80)); // abscisse
	rng.fill(trainData1.col(1), RNG::NORMAL, Scalar(150), Scalar(80)); // ordonnee
	rng.fill(trainData2.col(0), RNG::NORMAL, Scalar(300), Scalar(80));// abscisse
	rng.fill(trainData2.col(1), RNG::NORMAL, Scalar(190), Scalar(80));// ordonnee

	vconcat(trainData1, trainData2, trainDataGlobal);
	vconcat(TrainClasses1, TrainClasses2, TrainClasseG);
	

	//test
	int test_sample_count = 1000;
	Mat TestData1(test_sample_count, 2, CV_32FC1); //cv : type opencv, 32:bits, F: float, C: canal , 1 : nombre de canaux
	Mat TestData2(test_sample_count, 2, CV_32FC1);
	//ne sert a rien
	 //Mat TestClasses1 = Mat::zeros(test_sample_count, 1, CV_32FC1); //train our classe with a matrice of zeros with size(100,1)
	 //Mat TestClasses2 = Mat::ones(test_sample_count, 1, CV_32FC1);
	Mat TestDataGlobal;
	Mat TestClasseG;
	// permettant de faire un tirage aléatoire



	rng.fill(TestData1.col(0), RNG::NORMAL, Scalar(30), Scalar(50)); // abscisse
	rng.fill(TestData1.col(1), RNG::NORMAL, Scalar(190), Scalar(50)); // ordonnee
	rng.fill(TestData2.col(0), RNG::NORMAL, Scalar(240), Scalar(50));// abscisse
	rng.fill(TestData2.col(1), RNG::NORMAL, Scalar(100), Scalar(50));// ordonnee
	vconcat(TestData1, TestData2, TestDataGlobal);


	
	// Création SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC); //that can be used for n-class classification (n ≥ 2).
	svm->setKernel(SVM::RBF); //kernel type RBF
	//svm->setGamma(0.001);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6)); //Here we specify a maximum number
		//of iterationsand a tolerance error so we allow the algorithm to finish in 
		//less number of steps even if the optimal hyperplane has not been computed yet


	svm->trainAuto(trainDataGlobal, ml::ROW_SAMPLE, TrainClasseG); //trainAuto my SVM : it will choose optimal param's
	//alone, but it will take a long time to compute

	//Il faut donc faire varier li et col de 0 à la largeur et hauteur de l’image représentant l’espace pour 
	 //visualiser la frontière (on classe tous les points de l’espace discrétisé).
	for (int li = 0; li < ImgSVM.rows; li++)
	{
		for (int col = 0; col < ImgSVM.cols; col++)
		{
			Echantillon.at<float>(0) = (float)(col);
			Echantillon.at<float>(1) = (float)(li);
			//Pour estimer la réponse d’un Kppv en un point de coordonnée (li,col) de l’espace :
			float response = svm->predict(Echantillon);


			if (response == 1)
			{
				ImgSVM.at<Vec3b>(li, col) = Vec3b(180, 0, 0);
			}
			else
			{
				ImgSVM.at<Vec3b>(li, col) = Vec3b(0, 180, 0);
			}

		}
	}
	//Enfin pour afficher les échantillons d’origine et l’image résultante :
	for (int i = 0; i < train_sample_count; i++)
	{
		Point pt; //création d'un point
		pt.x = trainData1.at<float>(i, 0); //coordonées X
		pt.y = trainData1.at<float>(i, 1); // Y

		circle(ImgSVM, pt, 2, CV_RGB(255, 0, 0), FILLED); //dessiner un cerle rouge derayon 2 à partir du pt

		pt.x = trainData2.at<float>(i, 0);//autre point X
		pt.y = trainData2.at<float>(i, 1); // Y

		circle(ImgSVM, pt, 2, CV_RGB(0, 255, 0), FILLED); //autre cercle verte
	}
	//On donne à chaque point de l’image une couleur correspondant à la classe 
	//trouvée par le SVM.

	imshow("res", ImgSVM);
	waitKey(0);

	for (int i = 0; i < TestData1.rows; i++)
	{
		Mat res;
		Echantillon.at<float>(0) = TestData1.at<float>(i, 0);
		Echantillon.at<float>(1) = TestData1.at<float>(i, 1);
		// predict on majority of k PlusProche:

		float cl = svm->predict(Echantillon);
		if (cl == 0)
			Perf.TN++;
		else
			Perf.FP++;

	}
	for (int i = 0; i < TestData2.rows; i++)
	{
		Mat res;
		Echantillon.at<float>(0) = TestData2.at<float>(i, 0);
		Echantillon.at<float>(1) = TestData2.at<float>(i, 1);
		// predict on majority of k PlusProche:

		float cl = svm->predict(Echantillon);
		if (cl == 1)
			Perf.TP++;
		else
			Perf.FN++;

	}


	//struct Smetrics//retour var Smetrics


	SMetric Res = CalculMetric(Perf);

	Affichage.Format(L" METRICS \n  Accuracy = %f \n Precision= %f \n Recall = %f \n F_Score = %f ", Metric.Accuracy, Metric.F_Score, Metric.Precision, Metric.Recall);
	Performance.Format(L" \n Performance \n TP=%f \n TN=%f \n FP=%f \n FN=%f  ", Perf.TN, Perf.FN, Perf.TP, Perf.FP);
	UpdateData(false);
	int kk = 0;
}


void CDemoOpencv44Dlg::OnBnClickedRandomforest()
// Random Forest
{
	// INIT
	int li, col, accuracy;
	int train_sample_count = 100; //iteration
	Mat trainData1(train_sample_count, 2, CV_32FC1); //cv : type opencv, 32:bits, F: float, C: canal , 1 : nombre de canaux
	Mat trainData2(train_sample_count, 2, CV_32FC1);
	Mat TrainClasses1 = Mat::zeros(train_sample_count, 1, CV_32SC1); //train our classe with a matrice of zeros with size(100,1)
	Mat TrainClasses2 = Mat::ones(train_sample_count, 1, CV_32SC1);//méme chose mais avec matrice de 1
	Mat trainDataGlobal;
	Mat TrainClasseG;
	Mat ImgRTREES = Mat::zeros(500, 500, CV_8UC3); //Image sera une matrice de 0 (500,500) qui est notre espace
	Mat Echantillon(1, 2, CV_32FC1);
	RNG rng(12345); // permettant de faire un tirage aléatoire


	Mat Reponse, Dist; //afficher la reponse ? !
	rng.fill(trainData1.col(0), RNG::NORMAL, Scalar(50), Scalar(80)); // abscisse
	rng.fill(trainData1.col(1), RNG::NORMAL, Scalar(150), Scalar(80)); // ordonnee
	rng.fill(trainData2.col(0), RNG::NORMAL, Scalar(300), Scalar(80));// abscisse
	rng.fill(trainData2.col(1), RNG::NORMAL, Scalar(190), Scalar(80));// ordonnee

	vconcat(trainData1, trainData2, trainDataGlobal);
	vconcat(TrainClasses1, TrainClasses2, TrainClasseG);


	//test
	int test_sample_count = 1000;
	Mat TestData1(test_sample_count, 2, CV_32FC1); //cv : type opencv, 32:bits, F: float, C: canal , 1 : nombre de canaux
	Mat TestData2(test_sample_count, 2, CV_32FC1);
	//ne sert a rien
	 //Mat TestClasses1 = Mat::zeros(test_sample_count, 1, CV_32FC1); //train our classe with a matrice of zeros with size(100,1)
	 //Mat TestClasses2 = Mat::ones(test_sample_count, 1, CV_32FC1);
	Mat TestDataGlobal;
	Mat TestClasseG;
	// permettant de faire un tirage aléatoire



	rng.fill(TestData1.col(0), RNG::NORMAL, Scalar(30), Scalar(50)); // abscisse
	rng.fill(TestData1.col(1), RNG::NORMAL, Scalar(190), Scalar(50)); // ordonnee
	rng.fill(TestData2.col(0), RNG::NORMAL, Scalar(240), Scalar(50));// abscisse
	rng.fill(TestData2.col(1), RNG::NORMAL, Scalar(100), Scalar(50));// ordonnee
	vconcat(TestData1, TestData2, TestDataGlobal);



	// Création RandomForest
	 Ptr<RTrees> rtrees = RTrees::create();
	 //rtrees->RTrees::setActiveVarCount(10);
	 //rtrees->RTrees::setCalculateVarImportance(1);
	 rtrees->train(trainDataGlobal, ml::ROW_SAMPLE, TrainClasseG); //trainAuto my RF : it will choose optimal param's

	 //rtrees->RTrees::getActiveVarCount(); The size of the randomly selected subset of features at each tree node and that 
	 //are used to find the best split(s). If you set it to 0 then the size will be set to the square root of the total number of features. Default value is 0.
	
	 //rtrees->RTrees::getCalculateVarImportance();If true then variable importance will be calculated and then it can be retrieved by RTrees::getVarImportance.
	 //Default value is false.

	 //rtrees->RTrees::getOOBError();computed at the training stage when calcOOBError is set to true. If this flag was set to false, 0 is returned. The OOB error 
	 //is also scaled by sample weighting.
	 //rtrees->RTrees::getTermCriteria();



	//Il faut donc faire varier li et col de 0 à la largeur et hauteur de l’image représentant l’espace pour 
	 //visualiser la frontière (on classe tous les points de l’espace discrétisé).
	for (int li = 0; li < ImgRTREES.rows; li++)
	{
		for (int col = 0; col < ImgRTREES.cols; col++)
		{
			Echantillon.at<float>(0) = (float)(col);
			Echantillon.at<float>(1) = (float)(li);
			//Pour estimer la réponse d’un Rtrees en un point de coordonnée (li,col) de l’espace :
			float response = rtrees->predict(Echantillon);


			if (response == 1)
			{
				ImgRTREES.at<Vec3b>(li, col) = Vec3b(180, 0, 0);
			}
			else
			{
				ImgRTREES.at<Vec3b>(li, col) = Vec3b(0, 180, 0);
			}

		}
	}
	//Enfin pour afficher les échantillons d’origine et l’image résultante :
	for (int i = 0; i < train_sample_count; i++)
	{
		Point pt; //création d'un point
		pt.x = trainData1.at<float>(i, 0); //coordonées X
		pt.y = trainData1.at<float>(i, 1); // Y

		circle(ImgRTREES, pt, 2, CV_RGB(255, 0, 0), FILLED); //dessiner un cerle rouge derayon 2 à partir du pt

		pt.x = trainData2.at<float>(i, 0);//autre point X
		pt.y = trainData2.at<float>(i, 1); // Y

		circle(ImgRTREES, pt, 2, CV_RGB(0, 255, 0), FILLED); //autre cercle verte
	}
	//On donne à chaque point de l’image une couleur correspondant à la classe 
	//trouvée par le SVM.

	imshow("res", ImgRTREES);
	waitKey(0);

	for (int i = 0; i < TestData1.rows; i++)
	{
		Mat res;
		Echantillon.at<float>(0) = TestData1.at<float>(i, 0);
		Echantillon.at<float>(1) = TestData1.at<float>(i, 1);
		// predict on majority of k PlusProche:

		float cl = rtrees->predict(Echantillon);
		if (cl == 0)
			Perf.TN++;
		else
			Perf.FP++;

	}
	for (int i = 0; i < TestData2.rows; i++)
	{
		Mat res;
		Echantillon.at<float>(0) = TestData2.at<float>(i, 0);
		Echantillon.at<float>(1) = TestData2.at<float>(i, 1);
		// predict on majority of k PlusProche:

		float cl = rtrees->predict(Echantillon);
		if (cl == 1)
			Perf.TP++;
		else
			Perf.FN++;

	}


	//struct Smetrics//retour var Smetrics


	SMetric Res = CalculMetric(Perf);

	Affichage.Format(L" METRICS \n  Accuracy = %f \n Precision= %f \n Recall = %f \n F_Score = %f ", Metric.Accuracy, Metric.F_Score, Metric.Precision, Metric.Recall);
	Performance.Format(L" \n Performance \n TP=%f \n TN=%f \n FP=%f \n FN=%f  ", Perf.TN, Perf.FN, Perf.TP, Perf.FP);
	UpdateData(false);
	int kk = 0;
}




void CDemoOpencv44Dlg::OnBnClickedNn()
{ 
	//train data
 	Mat TrainData1;
	Mat TrainClasse1;
	Mat TrainData2;
	Mat TrainClasse2;
	GenData(TrainData1, TrainClasse1, 1, 100, 150, 80);
	GenData(TrainData2, TrainClasse2, 1, 320, 290, 80);
	Mat ImgNN = Mat::zeros(500, 500, CV_8UC3); //Image sera une matrice de 0 (500,500) qui est notre espace
	Mat Echantillon(1, 2, CV_32FC1);
	Mat trainDataGlobal;
	Mat TrainClasseG;
	vconcat(TrainData1, TrainData2, trainDataGlobal);
	vconcat(TrainClasse1, TrainClasse2, TrainClasseG);
	int train_sample_count = 100; //iteration
	RNG rng(12345); // permettant de faire un tirage aléatoire


	vconcat(TrainData1, TrainData2, trainDataGlobal);
	vconcat(TrainClasse1, TrainClasse2, TrainClasseG);

	//test data
	int test_sample_count = 1000;
	Mat TestData1;
	Mat TestData2;
	Mat TestClasse1;
	Mat TestClasse2;

	Mat TestDataGlobal;
	Mat TestClasseG;
	// permettant de faire un tirage aléatoire
	GenData(TestData1, TestClasse1, 1, 300, 190, 80);
	GenData(TestData2, TestClasse2, 1, 250, 240, 80);
	vconcat(TestData1, TestData2, TestDataGlobal);
	vconcat(TestClasse1, TestClasse2, TestDataGlobal);

	
	cv::Mat layers = cv::Mat(3, 1, CV_32FC1);//2 entrées et une couche de sortie
	layers.row(0) = cv::Scalar(2); //Couche d'entrée
	layers.row(1) = cv::Scalar(2);//couche sortie
	layers.row(2) = cv::Scalar(1);//couche sortie
	int nfeatures = TrainData1.cols;
	Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create(); //création
	ann->setLayerSizes(layers);// Mettre les layers en place
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM); // mettre en place la fct d'activation 
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.01)); //mettre a jour les critére : 300=epochs. 0.0001 = lr
	ann->setTrainMethod(ml::ANN_MLP::BACKPROP);// set backpropagation with lr=0.0001
	ann->train(trainDataGlobal, ml::ROW_SAMPLE, TrainClasseG);
	for (int li = 0; li < ImgNN.rows; li++)
	{
		for(int col=0 ; col < ImgNN.cols ; col++)
		{ 
			Echantillon.at<float>(0) = (float)(col);
			Echantillon.at<float>(1) = (float)(li);
			//Pour estimer la réponse d’un NN en un point de coordonnée (li,col) de l’espace :
			float response = ann->predict(Echantillon);
			//Mat result;
			//float response = ann->predict(Mat::ones(1, trainDataGlobal.cols, trainDataGlobal.type()), result);

			if (response == 1)
			{
				ImgNN.at<Vec3b>(li, col) = Vec3b(180, 0, 0);
			}
			else
			{
				ImgNN.at<Vec3b>(li, col) = Vec3b(0, 180, 0);
			}

		}
	}
	//Enfin pour afficher les échantillons d’origine et l’image résultante :
	for (int i = 0; i < train_sample_count; i++)
	{
		Point pt; //création d'un point
		pt.x = TestData1.at<float>(i, 0); //coordonées X
		pt.y = TestData1.at<float>(i, 1); // Y

		circle(ImgNN, pt, 2, CV_RGB(0, 0, 255), FILLED); //dessiner un cerle rouge derayon 2 à partir du pt

		pt.x = TestData1.at<float>(i, 0);//autre point X
		pt.y = TestData1.at<float>(i, 1); // Y

		circle(ImgNN, pt, 2, CV_RGB(0, 0, 0), FILLED); //autre cercle verte
	}
	//On donne à chaque point de l’image une couleur correspondant à la classe 
	//trouvée par le SVM.

	imshow("res", ImgNN);
	waitKey(0);

	for (int i = 0; i < TestData1.rows; i++)
	{
		Mat res;
		Echantillon.at<float>(0) = TestData1.at<float>(i, 0);
		Echantillon.at<float>(1) = TestData1.at<float>(i, 1);
		// predict on majority of k PlusProche:

		float cl = ann->predict(Echantillon);
		if (cl == 0)
			Perf.TN++;
		else
			Perf.FP++;

	}
	for (int i = 0; i < TestData2.rows; i++)
	{
		Mat res;
		Echantillon.at<float>(0) = TestData2.at<float>(i, 0);
		Echantillon.at<float>(1) = TestData2.at<float>(i, 1);
		// predict on majority of k PlusProche:

		float cl = ann->predict(Echantillon);
		if (cl == 1)
			Perf.TP++;
		else
			Perf.FN++;

	}


	//struct Smetrics//retour var Smetrics


	SMetric Res = CalculMetric(Perf);

	Affichage.Format(L" METRICS \n  Accuracy = %f \n Precision= %f \n Recall = %f \n F_Score = %f ", Metric.Accuracy, Metric.F_Score, Metric.Precision, Metric.Recall);
	Performance.Format(L" \n Performance \n TP=%f \n TN=%f \n FP=%f \n FN=%f  ", Perf.TN, Perf.FN, Perf.TP, Perf.FP);
	UpdateData(false);
	int kk = 0;
}
