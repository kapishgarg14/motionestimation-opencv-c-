#include "features.h"
#include<iostream>
using namespace cv;
using namespace std;

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000
int main( )	{

  Mat img_1, img_2;
  Mat R_f, t_f; 

  double scale = 1.00;
  char text[100];
  int fontFace = FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;  
  cv::Point textOrg(10, 50);

  //read the first two frames from the dataset
  Mat img_1_c = imread( "data/0.png");
  Mat img_2_c = imread( "data/1.png");

  // we work with grayscale images
  cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
  cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

  // feature detection, tracking
  vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
  featureDetection(img_1, points1);        //detect features in img_1
  vector<uchar> status;
  featureTracking(img_1,img_2,points1,points2, status); //track those features to img_2
  double focal = 7.215377e+02;
  cv::Point2d pp(6.095593e+02,1.728540e+02);
  Mat E, R, t, mask;
  E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
  recoverPose(E, points2, points1, R, t, focal, pp, mask);

  Mat prevImage = img_2;
  Mat currImage;
  vector<Point2f> prevFeatures = points2;
  vector<Point2f> currFeatures;
  clock_t begin = clock();

  namedWindow( "Road facing camera", WINDOW_AUTOSIZE );
  namedWindow( "Trajectory", WINDOW_AUTOSIZE );

  Mat traj = Mat::zeros(600, 600, CV_8UC3);
  Mat intialpos=Mat::eye(Size(4,4),R.type());


  for(int numFrame=2; numFrame < 447; numFrame++)	{
  	Mat currImage_c = imread("data/"+to_string(numFrame)+".png");
  	cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
  	vector<uchar> status;
  	featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

  	E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
  	recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

    Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2,currFeatures.size(), CV_64F);


   for(int i=0;i<prevFeatures.size();i++)	{   
  		prevPts.at<double>(0,i) = prevFeatures.at(i).x;
  		prevPts.at<double>(1,i) = prevFeatures.at(i).y;

  		currPts.at<double>(0,i) = currFeatures.at(i).x;
  		currPts.at<double>(1,i) = currFeatures.at(i).y;
    }


   Mat invTrans=R.t();
   Mat l=-R.t()*t;
   hconcat(invTrans,l,invTrans);
   double k[4]={0,0,0,1};
   Mat bottom(1,4,invTrans.type(),k);
   vconcat(invTrans,bottom,invTrans);
   intialpos=intialpos*invTrans;

 	  if (prevFeatures.size() < MIN_NUM_FEAT)	{
 		  featureDetection(prevImage, prevFeatures);
      featureTracking(prevImage,currImage,prevFeatures,currFeatures, status);
 	  }

    prevImage = currImage.clone();
    prevFeatures = currFeatures;

    int x = int(intialpos.at<double>(0,3)) +300;
    int y = int(intialpos.at<double>(2,3)) +500;
    circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);

    rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0),-1);
    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", intialpos.at<double>(0,3), intialpos.at<double>(1,3), intialpos.at<double>(2,3));
    putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    imshow( "Road facing camera",currImage_c );
    imshow( "Trajectory",traj );

    waitKey(1);

  }

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Total time taken: " << elapsed_secs << "s" << endl;

  return 0;
}
/* 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 
0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00
 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00*/
