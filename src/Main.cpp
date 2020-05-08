#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, const char* argv[])
{
	//load images
    cv::Mat img_1 = cv::imread("./IMG_CAL_DATA/left08.png", 0);
    cv::Mat img_2 = cv::imread("./IMG_CAL_DATA/right08.png", 0);

    //create feature detectors & descriptor extractors
    int minHessian = 2000; //filter parameter
  	cv::Ptr<SURF> SURF = SURF::create(minHessian);
  	double contrastThreshold = 0.99; //filter parameter
  	double edgeThreshold = 2; //filter parameter
  	Ptr<SIFT> SIFT = SIFT::create(contrastThreshold, edgeThreshold);
  	cv::Ptr<Feature2D> ORB = ORB::create();

  	//detect keypoints
  	std::vector<KeyPoint> keypoints_1, keypoints_2;    
  	SURF->detect( img_1, keypoints_1 );
  	SURF->detect( img_2, keypoints_2 );

  	//calculate descriptors 
 	Mat descriptors_1, descriptors_2;    
  	SURF->compute( img_1, keypoints_1, descriptors_1 );
  	SURF->compute( img_2, keypoints_2, descriptors_2 );

  	//match descriptors using BFMatcher
  	int normType = NORM_L1; //should be different for each feature detector, but works well
  	bool crossCheck = true; //match test
  	Ptr<BFMatcher> matcher = BFMatcher::create(normType, crossCheck);
  	std::vector< DMatch > matches;
  	matcher->match( descriptors_1, descriptors_2, matches );

  	//draw results to image
  	Mat img_matches;
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  	//save image
    cv::imwrite("LR-SURF.png", img_matches);

    //detect keypoints
  	SIFT->detect( img_1, keypoints_1 );
  	SIFT->detect( img_2, keypoints_2 );

  	//calculate descriptors 
  	SIFT->compute( img_1, keypoints_1, descriptors_1 );
  	SIFT->compute( img_2, keypoints_2, descriptors_2 );

  	//match descriptors using BFMatcher
  	matcher->match( descriptors_1, descriptors_2, matches );

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  	//save image
    cv::imwrite("LR-SIFT.png", img_matches);

    //detect keypoints
  	ORB->detect( img_1, keypoints_1 );
  	ORB->detect( img_2, keypoints_2 );

  	//calculate descriptors
  	ORB->compute( img_1, keypoints_1, descriptors_1 );
  	ORB->compute( img_2, keypoints_2, descriptors_2 );

  	//match using BFMatcher
  	matcher->match( descriptors_1, descriptors_2, matches );

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  	//save image
    cv::imwrite("LR-ORB.png", img_matches);

    //detect keypoints
    int treshold = 120; //filter parameter
    cv::FAST(img_1, keypoints_1, treshold, true);
    cv::FAST(img_2, keypoints_2, treshold, true);

    //create descriptor extractor and calculate descriptors
    Ptr<DescriptorExtractor> featureExtractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    featureExtractor->compute(img_1, keypoints_1, descriptors_1);
    featureExtractor->compute(img_2, keypoints_2, descriptors_2);

    //match descriptors using BFMatcher
  	matcher->match( descriptors_1, descriptors_2, matches );

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  	//save image
    cv::imwrite("LR-FAST+BRIEF.png", img_matches);

	//load images
    img_2 = cv::imread("./IMG_CAL_DATA/left10.png", 0);

  	//detect keypoints   
  	SURF->detect( img_1, keypoints_1 );
  	SURF->detect( img_2, keypoints_2 );

  	//calculate descriptors 
  	SURF->compute( img_1, keypoints_1, descriptors_1 );
  	SURF->compute( img_2, keypoints_2, descriptors_2 );

  	//match descriptors using BFMatcher
  	matcher->match( descriptors_1, descriptors_2, matches );

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  	//save image
    cv::imwrite("LL-SURF.png", img_matches);

    //detect keypoints
  	SIFT->detect( img_1, keypoints_1 );
  	SIFT->detect( img_2, keypoints_2 );

  	//calculate descriptors 
  	SIFT->compute( img_1, keypoints_1, descriptors_1 );
  	SIFT->compute( img_2, keypoints_2, descriptors_2 );

  	//match descriptors using BFMatcher
  	matcher->match( descriptors_1, descriptors_2, matches );

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  	//save image
    cv::imwrite("LL-SIFT.png", img_matches);

    //detect keypoints
  	ORB->detect( img_1, keypoints_1 );
  	ORB->detect( img_2, keypoints_2 );

  	//calculate descriptors
  	ORB->compute( img_1, keypoints_1, descriptors_1 );
  	ORB->compute( img_2, keypoints_2, descriptors_2 );

  	//match using BFMatcher
  	matcher->match( descriptors_1, descriptors_2, matches );

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  	//save image
    cv::imwrite("LL-ORB.png", img_matches);

    //detect keypoints
    cv::FAST(img_1, keypoints_1, treshold, true);
    cv::FAST(img_2, keypoints_2, treshold, true);

    //calculate descriptors
	featureExtractor->compute(img_1, keypoints_1, descriptors_1);
    featureExtractor->compute(img_2, keypoints_2, descriptors_2);

    //match descriptors using BFMatcher
  	matcher->match( descriptors_1, descriptors_2, matches );

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  	//save image
    cv::imwrite("LL-FAST+BRIEF.png", img_matches);

  	return 0;
}