#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>

using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, const char* argv[])
{
	//load images
    cv::Mat img_1 = cv::imread("./IMG_CAL_DATA/left08.png", 0);
    cv::Mat img_2 = cv::imread("./IMG_CAL_DATA/right08.png", 0);

  	cv::Ptr<SURF> SURF = SURF::create(); //SURF feature extractor
  	Ptr<SIFT> SIFT = SIFT::create(); //SIFT feature extractor
  	cv::Ptr<Feature2D> ORB = ORB::create(); //ORB feature extractor
  	std::vector<KeyPoint> keypoints_1, keypoints_2; //keypoints vectors
  	Mat img_keypoints1, img_keypoints2; //img matrix for the keypoints
  	Mat descriptors_1, descriptors_2; //descriptors
  	int normType = NORM_L2; //should be different for each feature detector, but works well on all of them with L1
  	Ptr<BFMatcher> matcher = BFMatcher::create(normType); //BF matcher
  	std::vector< std::vector<DMatch >> allMatches; //matches vector
  	double ratio = 0.6; //ratio test
  	Mat img_matches; //img showing matches
  	int noOfLines = 40; //number of epipolar lines
  	std::vector<Point2f> points1(noOfLines); //points of the 1st image for epipolar lines
	std::vector<Point2f> points2(noOfLines); //points of the 2nd image for epipolar lines
	std::vector< DMatch > epipolarMatches; //matches used for the drawing of epipolar lines
	int point_count = 15; //points used for FM & EM estimation
	std::vector<Point2f> points1fm(point_count); //points for the FM estimation
	std::vector<Point2f> points2fm(point_count); //points for the FM estimation
	Mat fundamental_matrix; //fundamental matrix
	Mat essential_matrix; //essential matrix
	Mat epipolarLines; //epipolar vector lines
	Mat epipolar_img; //image with epipolar lines
	cv::RNG rng(0); //random color generator
	Mat img_1rgb; //complemental image with the same colors as epipolar_img for concatenation

	double kl[3][3] = {{9.842439e+02, 0.000000e+00, 6.900000e+02}, {0.000000e+00, 9.808141e+02, 2.331966e+02}, {0.000000e+00, 0.000000e+00, 1.000000e+00}};
	Mat KL = Mat(3, 3, CV_64F, kl);
	double kr[3][3] = {{9.895267e+02, 0.000000e+00, 7.020000e+02}, {0.000000e+00, 9.878386e+02, 2.455590e+02}, {0.000000e+00, 0.000000e+00, 1.000000e+00}};
	Mat KR = Mat(3, 3, CV_64F, kr);
	Mat Kavg = (KL+KR)*0.5;

  	//-----------------------------------------SURF LR--------------------------------------------------------//
  	//detect keypoints  
  	SURF->detect( img_1, keypoints_1 );
  	SURF->detect( img_2, keypoints_2 );

    drawKeypoints( img_1, keypoints_1, img_keypoints1 );
    //-- Show detected (drawn) keypoints
    cv::imwrite("left08-SURF.png", img_keypoints1);
    drawKeypoints( img_2, keypoints_2, img_keypoints2 );
    //-- Show detected (drawn) keypoints
    cv::imwrite("right08-SURF.png", img_keypoints2);
    
  	//calculate descriptors   
  	SURF->compute( img_1, keypoints_1, descriptors_1 );
  	SURF->compute( img_2, keypoints_2, descriptors_2 );

  	//match descriptors using BFMatcher
  	matcher->knnMatch( descriptors_1, descriptors_2, allMatches, 2);
  	std::vector<DMatch> matches;
  	ratio = 0.2;         
    for(int i = 0; i < allMatches.size(); i++){  
        if(allMatches[i][0].distance < ratio * allMatches[i][1].distance)
            matches.push_back(allMatches[i][0]);
    }

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  	//save image
    cv::imwrite("LR-SURF.png", img_matches);

	// initialize the points for the epipolar lines ... */
	for( int i = 0; i < noOfLines; i++ )
	{
	    points1[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2[i] = keypoints_2[matches[i].trainIdx].pt;
	    epipolarMatches.push_back(matches[i]);

	}

	// initialize the points for the fundamental matrix calculation ... */
	for( int i = 0; i < point_count; i++ )
	{
	    points1fm[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2fm[i] = keypoints_2[matches[i].trainIdx].pt;

	}

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_8POINT);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file("LR-SURF-8pt.txt", cv::FileStorage::WRITE);
	file << "Fundamental Matrix" << fundamental_matrix;
	file << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	cv::Scalar color(rng(256),rng(256),rng(256));
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2fm[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LR-SURF-epipolar-8pt.png", epipolar_img);

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_RANSAC);
	essential_matrix = KR.t()*fundamental_matrix*KL;
	

	//output to text file
	cv::FileStorage file1("LR-SURF-RANSAC.txt", cv::FileStorage::WRITE);
	file1 << "Fundamental Matrix" << fundamental_matrix;
	file1 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LR-SURF-epipolar-RANSAC.png", epipolar_img);

    essential_matrix = cv::findEssentialMat(points1fm, points2fm, Kavg, RANSAC);
    fundamental_matrix = KR.t().inv()*essential_matrix*KL.inv();

    //output to text file
	cv::FileStorage file16("LR-SURF-5pt.txt", cv::FileStorage::WRITE);
	file16 << "Fundamental Matrix" << fundamental_matrix;
	file16 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LR-SURF-epipolar-5pt.png", epipolar_img);

    std::cout << "[DEBUG] Finished LR SURF" << std::endl;
    std::cout.flush();

    //-----------------------------------------------------SIFT LR--------------------------------------------------//
    //detect keypoints
  	SIFT->detect( img_1, keypoints_1 );
  	SIFT->detect( img_2, keypoints_2 );

    drawKeypoints( img_1, keypoints_1, img_keypoints1 );
    //-- Show detected (drawn) keypoints
    cv::imwrite("left08-SIFT.png", img_keypoints1);
    drawKeypoints( img_2, keypoints_2, img_keypoints2 );
    //-- Show detected (drawn) keypoints
    cv::imwrite("right08-SIFT.png", img_keypoints2);
    
  	//calculate descriptors 
  	SIFT->compute( img_1, keypoints_1, descriptors_1 );
  	SIFT->compute( img_2, keypoints_2, descriptors_2 );

  	//match descriptors using BFMatcher
  	allMatches.clear();
  	matcher->knnMatch( descriptors_1, descriptors_2, allMatches, 2);
  	matches.clear();
  	ratio = 0.3;         
    for(int i = 0; i < allMatches.size(); i++){  
        if(allMatches[i][0].distance < ratio * allMatches[i][1].distance)
            matches.push_back(allMatches[i][0]);
    }

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  	//save image
    cv::imwrite("LR-SIFT.png", img_matches);

	// initialize the points for the epipolar lines ... */
	for( int i = 0; i < noOfLines; i++ )
	{
	    points1[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2[i] = keypoints_2[matches[i].trainIdx].pt;
	    epipolarMatches.push_back(matches[i]);

	}
	// initialize the points for the fundamental matrix calculation ... */
	for( int i = 0; i < point_count; i++ )
	{
	    points1fm[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2fm[i] = keypoints_2[matches[i].trainIdx].pt;

	}

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_8POINT);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file2("LR-SIFT-8pt.txt", cv::FileStorage::WRITE);
	file2 << "Fundamental Matrix" << fundamental_matrix;
	file2 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR);
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LR-SIFT-epipolar-8pt.png", epipolar_img);

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_RANSAC);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file3("LR-SIFT-RANSAC.txt", cv::FileStorage::WRITE);
	file3 << "Fundamental Matrix" << fundamental_matrix;
	file3 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LR-SIFT-epipolar-RANSAC.png", epipolar_img);

    essential_matrix = cv::findEssentialMat(points1fm, points2fm, Kavg, RANSAC);
    fundamental_matrix = KR.t().inv()*essential_matrix*KL.inv();

    //output to text file
	cv::FileStorage file17("LR-SIFT-5pt.txt", cv::FileStorage::WRITE);
	file17 << "Fundamental Matrix" << fundamental_matrix;
	file17 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LR-SIFT-epipolar-5pt.png", epipolar_img);

    std::cout << "[DEBUG] Finished LR SIFT" << std::endl;
    std::cout.flush();

    //------------------------------------------------------ORB LR---------------------------------------------------//
    //detect keypoints
  	ORB->detect( img_1, keypoints_1 );
  	ORB->detect( img_2, keypoints_2 );

    drawKeypoints( img_1, keypoints_1, img_keypoints1 );
    //-- Show detected (drawn) keypoints
    cv::imwrite("left08-ORB.png", img_keypoints1);
    drawKeypoints( img_2, keypoints_2, img_keypoints2 );
    //-- Show detected (drawn) keypoints
    cv::imwrite("right08-ORB.png", img_keypoints2);

  	//calculate descriptors
  	ORB->compute( img_1, keypoints_1, descriptors_1 );
  	ORB->compute( img_2, keypoints_2, descriptors_2 );

  	//match descriptors using BFMatcher
  	allMatches.clear();
  	matcher->knnMatch( descriptors_1, descriptors_2, allMatches, 2);
  	matches.clear();
  	ratio = 0.6;     
    for(int i = 0; i < allMatches.size(); i++){  
        if(allMatches[i][0].distance < ratio * allMatches[i][1].distance)
            matches.push_back(allMatches[i][0]);
    }

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  	//save image
    cv::imwrite("LR-ORB.png", img_matches);

    // initialize the points for the epipolar lines ... */
	for( int i = 0; i < noOfLines; i++ )
	{
	    points1[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2[i] = keypoints_2[matches[i].trainIdx].pt;
	    epipolarMatches.push_back(matches[i]);

	}
	// initialize the points for the fundamental matrix calculation ... */
	for( int i = 0; i < point_count; i++ )
	{
	    points1fm[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2fm[i] = keypoints_2[matches[i].trainIdx].pt;

	}

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_8POINT);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file4("LR-ORB-8pt.txt", cv::FileStorage::WRITE);
	file4 << "Fundamental Matrix" << fundamental_matrix;
	file4 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR);
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LR-ORB-epipolar-8pt.png", epipolar_img);

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_RANSAC);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file5("LR-ORB-RANSAC.txt", cv::FileStorage::WRITE);
	file5 << "Fundamental Matrix" << fundamental_matrix;
	file5 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LR-ORB-epipolar-RANSAC.png", epipolar_img);

    essential_matrix = cv::findEssentialMat(points1fm, points2fm, Kavg, RANSAC);
    fundamental_matrix = KR.t().inv()*essential_matrix*KL.inv();

    //output to text file
	cv::FileStorage file18("LR-ORB-5pt.txt", cv::FileStorage::WRITE);
	file18 << "Fundamental Matrix" << fundamental_matrix;
	file18 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LR-ORB-epipolar-5pt.png", epipolar_img);

    std::cout << "[DEBUG] Finished LR ORB" << std::endl;
    std::cout.flush();

    //------------------------------------------------------FAST+BRIEF LR------------------------------------------------//
    //detect keypoints
    int treshold = 120; //filter parameter
    cv::FAST(img_1, keypoints_1, treshold, true);
    cv::FAST(img_2, keypoints_2, treshold, true);

    drawKeypoints( img_1, keypoints_1, img_keypoints1 );
    //-- Show detected (drawn) keypoints
    cv::imwrite("left08-FAST.png", img_keypoints1);
    drawKeypoints( img_2, keypoints_2, img_keypoints2 );
    //-- Show detected (drawn) keypoints
    cv::imwrite("right08-FAST.png", img_keypoints2);

    //create descriptor extractor and calculate descriptors
    Ptr<DescriptorExtractor> featureExtractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    featureExtractor->compute(img_1, keypoints_1, descriptors_1);
    featureExtractor->compute(img_2, keypoints_2, descriptors_2);

    //match descriptors using BFMatcher
  	allMatches.clear();
  	matcher->knnMatch( descriptors_1, descriptors_2, allMatches, 2);
  	matches.clear();
  	ratio = 0.5;     
    for(int i = 0; i < allMatches.size(); i++){  
        if(allMatches[i][0].distance < ratio * allMatches[i][1].distance)
            matches.push_back(allMatches[i][0]);
    }

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  	//save image
    cv::imwrite("LR-FAST+BRIEF.png", img_matches);

	// initialize the points for the epipolar lines ... */
	for( int i = 0; i < noOfLines; i++ )
	{
	    points1[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2[i] = keypoints_2[matches[i].trainIdx].pt;
	    epipolarMatches.push_back(matches[i]);

	}
	// initialize the points for the fundamental matrix calculation ... */
	for( int i = 0; i < point_count; i++ )
	{
	    points1fm[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2fm[i] = keypoints_2[matches[i].trainIdx].pt;

	}

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_8POINT);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file6("LR-FAST-8pt.txt", cv::FileStorage::WRITE);
	file6 << "Fundamental Matrix" << fundamental_matrix;
	file6 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR);
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LR-FAST-epipolar-8pt.png", epipolar_img);

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_RANSAC);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file7("LR-FAST-RANSAC.txt", cv::FileStorage::WRITE);
	file7 << "Fundamental Matrix" << fundamental_matrix;
	file7 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LR-FAST-epipolar-RANSAC.png", epipolar_img);

    essential_matrix = cv::findEssentialMat(points1fm, points2fm, Kavg, RANSAC);
    fundamental_matrix = KR.t().inv()*essential_matrix*KL.inv();

    //output to text file
	cv::FileStorage file19("LR-FAST-5pt.txt", cv::FileStorage::WRITE);
	file19 << "Fundamental Matrix" << fundamental_matrix;
	file19 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LR-FAST-epipolar-5pt.png", epipolar_img);

    std::cout << "[DEBUG] Finished LR FAST" << std::endl;
    std::cout.flush();

    //------------------------------------------------------LL-----------------------------------------------------//
	//load images
    img_2 = cv::imread("./IMG_CAL_DATA/left10.png", 0);

    //----------------------------------------------------------SURF LL----------------------------------------------------//
  	//detect keypoints   
  	SURF->detect( img_1, keypoints_1 );
  	SURF->detect( img_2, keypoints_2 );

  	drawKeypoints( img_2, keypoints_2, img_keypoints2 );
    //-- Show detected (drawn) keypoints
    cv::imwrite("left10-SURF.png", img_keypoints2);

  	//calculate descriptors 
  	SURF->compute( img_1, keypoints_1, descriptors_1 );
  	SURF->compute( img_2, keypoints_2, descriptors_2 );

  	//match descriptors using BFMatcher
  	allMatches.clear();
  	matcher->knnMatch( descriptors_1, descriptors_2, allMatches, 2);
  	matches.clear();   
  	ratio = 0.2;      
    for(int i = 0; i < allMatches.size(); i++){  
        if(allMatches[i][0].distance < ratio * allMatches[i][1].distance)
            matches.push_back(allMatches[i][0]);
    }

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  	//save image
    cv::imwrite("LL-SURF.png", img_matches);

	// initialize the points for the epipolar lines ... */
	for( int i = 0; i < noOfLines; i++ )
	{
	    points1[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2[i] = keypoints_2[matches[i].trainIdx].pt;
	    epipolarMatches.push_back(matches[i]);

	}

	// initialize the points for the fundamental matrix calculation ... */
	for( int i = 0; i < point_count; i++ )
	{
	    points1fm[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2fm[i] = keypoints_2[matches[i].trainIdx].pt;

	}

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_8POINT);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file8("LL-SURF-8pt.txt", cv::FileStorage::WRITE);
	file8 << "Fundamental Matrix" << fundamental_matrix;
	file8 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2fm[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LL-SURF-epipolar-8pt.png", epipolar_img);

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_RANSAC);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file9("LL-SURF-RANSAC.txt", cv::FileStorage::WRITE);
	file9 << "Fundamental Matrix" << fundamental_matrix;
	file9 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LL-SURF-epipolar-RANSAC.png", epipolar_img);

    essential_matrix = cv::findEssentialMat(points1fm, points2fm, Kavg, RANSAC);
    fundamental_matrix = KR.t().inv()*essential_matrix*KL.inv();

    //output to text file
	cv::FileStorage file20("LL-SURF-5pt.txt", cv::FileStorage::WRITE);
	file20 << "Fundamental Matrix" << fundamental_matrix;
	file20 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LL-SURF-epipolar-5pt.png", epipolar_img);

    std::cout << "[DEBUG] Finished LL SURF" << std::endl;
    std::cout.flush();

    //----------------------------------------------------------SIFT LL----------------------------------------------------//
    //detect keypoints
  	SIFT->detect( img_1, keypoints_1 );
  	SIFT->detect( img_2, keypoints_2 );

  	drawKeypoints( img_2, keypoints_2, img_keypoints2 );
    //-- Show detected (drawn) keypoints
    cv::imwrite("left10-SIFT.png", img_keypoints2);

  	//calculate descriptors 
  	SIFT->compute( img_1, keypoints_1, descriptors_1 );
  	SIFT->compute( img_2, keypoints_2, descriptors_2 );

  	//match descriptors using BFMatcher
  	allMatches.clear();
  	matcher->knnMatch( descriptors_1, descriptors_2, allMatches, 2);
  	matches.clear();   
  	ratio = 0.2;      
    for(int i = 0; i < allMatches.size(); i++){  
        if(allMatches[i][0].distance < ratio * allMatches[i][1].distance)
            matches.push_back(allMatches[i][0]);
    }

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  	//save image
    cv::imwrite("LL-SIFT.png", img_matches);

	// initialize the points for the epipolar lines ... */
	for( int i = 0; i < noOfLines; i++ )
	{
	    points1[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2[i] = keypoints_2[matches[i].trainIdx].pt;
	    epipolarMatches.push_back(matches[i]);

	}

	// initialize the points for the fundamental matrix calculation ... */
	for( int i = 0; i < point_count; i++ )
	{
	    points1fm[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2fm[i] = keypoints_2[matches[i].trainIdx].pt;

	}

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_8POINT);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file10("LL-SIFT-8pt.txt", cv::FileStorage::WRITE);
	file10 << "Fundamental Matrix" << fundamental_matrix;
	file10 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2fm[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LL-SIFT-epipolar-8pt.png", epipolar_img);

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_RANSAC);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file11("LL-SIFT-RANSAC.txt", cv::FileStorage::WRITE);
	file11 << "Fundamental Matrix" << fundamental_matrix;
	file11 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LL-SIFT-epipolar-RANSAC.png", epipolar_img);

    essential_matrix = cv::findEssentialMat(points1fm, points2fm, Kavg, RANSAC);
    fundamental_matrix = KR.t().inv()*essential_matrix*KL.inv();

    //output to text file
	cv::FileStorage file21("LL-SIFT-5pt.txt", cv::FileStorage::WRITE);
	file21 << "Fundamental Matrix" << fundamental_matrix;
	file21 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LL-SIFT-epipolar-5pt.png", epipolar_img);

    std::cout << "[DEBUG] Finished LL SIFT" << std::endl;
    std::cout.flush();

    //----------------------------------------------------------ORB LL----------------------------------------------------//
    //detect keypoints
  	ORB->detect( img_1, keypoints_1 );
  	ORB->detect( img_2, keypoints_2 );

  	drawKeypoints( img_2, keypoints_2, img_keypoints2 );
    //-- Show detected (drawn) keypoints
    cv::imwrite("left10-ORB.png", img_keypoints2);

  	//calculate descriptors
  	ORB->compute( img_1, keypoints_1, descriptors_1 );
  	ORB->compute( img_2, keypoints_2, descriptors_2 );

  	//match descriptors using BFMatcher
  	allMatches.clear();
  	matcher->knnMatch( descriptors_1, descriptors_2, allMatches, 2);
  	matches.clear();   
  	ratio = 0.8;      
    for(int i = 0; i < allMatches.size(); i++){  
        if(allMatches[i][0].distance < ratio * allMatches[i][1].distance)
            matches.push_back(allMatches[i][0]);
    }

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  	//save image
    cv::imwrite("LL-ORB.png", img_matches);

	// initialize the points for the epipolar lines ... */
	for( int i = 0; i < noOfLines; i++ )
	{
	    points1[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2[i] = keypoints_2[matches[i].trainIdx].pt;
	    epipolarMatches.push_back(matches[i]);

	}

	// initialize the points for the fundamental matrix calculation ... */
	for( int i = 0; i < point_count; i++ )
	{
	    points1fm[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2fm[i] = keypoints_2[matches[i].trainIdx].pt;

	}

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_8POINT);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file12("LL-ORB-8pt.txt", cv::FileStorage::WRITE);
	file12 << "Fundamental Matrix" << fundamental_matrix;
	file12 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2fm[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LL-ORB-epipolar-8pt.png", epipolar_img);

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_RANSAC);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file13("LL-ORB-RANSAC.txt", cv::FileStorage::WRITE);
	file13 << "Fundamental Matrix" << fundamental_matrix;
	file13 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LL-ORB-epipolar-RANSAC.png", epipolar_img);

    essential_matrix = cv::findEssentialMat(points1fm, points2fm, Kavg, RANSAC);
    fundamental_matrix = KR.t().inv()*essential_matrix*KL.inv();

    //output to text file
	cv::FileStorage file22("LL-ORB-5pt.txt", cv::FileStorage::WRITE);
	file22 << "Fundamental Matrix" << fundamental_matrix;
	file22 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LL-ORB-epipolar-5pt.png", epipolar_img);

    std::cout << "[DEBUG] Finished LL ORB" << std::endl;
    std::cout.flush();

    //----------------------------------------------------------FAST+BRIEF LL--------------------------------------------//
    //detect keypoints
    cv::FAST(img_1, keypoints_1, treshold, true);
    cv::FAST(img_2, keypoints_2, treshold, true);

    drawKeypoints( img_2, keypoints_2, img_keypoints2 );
    //-- Show detected (drawn) keypoints
    cv::imwrite("left10-FAST.png", img_keypoints2);

    //calculate descriptors
	featureExtractor->compute(img_1, keypoints_1, descriptors_1);
    featureExtractor->compute(img_2, keypoints_2, descriptors_2);

    //match descriptors using BFMatcher
  	allMatches.clear();
  	matcher->knnMatch( descriptors_1, descriptors_2, allMatches, 2);
  	matches.clear();   
  	ratio = 0.8;      
    for(int i = 0; i < allMatches.size(); i++){  
        if(allMatches[i][0].distance < ratio * allMatches[i][1].distance)
            matches.push_back(allMatches[i][0]);
    }

  	//draw results to image
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  	//save image
    cv::imwrite("LL-FAST+BRIEF.png", img_matches);

	// initialize the points for the epipolar lines ... */
	for( int i = 0; i < noOfLines; i++ )
	{
	    points1[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2[i] = keypoints_2[matches[i].trainIdx].pt;
	    epipolarMatches.push_back(matches[i]);

	}

	// initialize the points for the fundamental matrix calculation ... */
	for( int i = 0; i < noOfLines; i++ )
	{
	    points1fm[i] = keypoints_1[matches[i].queryIdx].pt;
	    points2fm[i] = keypoints_2[matches[i].trainIdx].pt;

	}

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_8POINT);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file14("LL-FAST-8pt.txt", cv::FileStorage::WRITE);
	file14 << "Fundamental Matrix" << fundamental_matrix;
	file14 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2fm[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LL-FAST-epipolar-8pt.png", epipolar_img);

	fundamental_matrix = findFundamentalMat(points1fm, points2fm, FM_RANSAC);
	essential_matrix = KR.t()*fundamental_matrix*KL;

	//output to text file
	cv::FileStorage file15("LL-FAST-RANSAC.txt", cv::FileStorage::WRITE);
	file15 << "Fundamental Matrix" << fundamental_matrix;
	file15 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LL-FAST-epipolar-RANSAC.png", epipolar_img);

    essential_matrix = cv::findEssentialMat(points1fm, points2fm, Kavg, RANSAC);
    fundamental_matrix = KR.t().inv()*essential_matrix*KL.inv();

    //output to text file
	cv::FileStorage file23("LL-FAST-5pt.txt", cv::FileStorage::WRITE);
	file23 << "Fundamental Matrix" << fundamental_matrix;
	file23 << "Essential Matrix" << essential_matrix;

	//calculate epipolar lines of right using left points
	cv::computeCorrespondEpilines(points1, 1, fundamental_matrix, epipolarLines);

	//prepare output image
	epipolar_img = img_2.clone();
	cvtColor(epipolar_img, epipolar_img, COLOR_GRAY2BGR); 
	
	//draw epipolar lines and keypoints of the right image
	for( int i = 0; i < noOfLines; i++ )
	{
	    line(epipolar_img,
      	cv::Point(0,-epipolarLines.at<Point3f>(i, 0).z/epipolarLines.at<Point3f>(i, 0).y),
      	cv::Point(epipolar_img.cols,-(epipolarLines.at<Point3f>(i, 0).z+epipolarLines.at<Point3f>(i, 0).x*epipolar_img.cols)/epipolarLines.at<Point3f>(i, 0).y),
      	color);
    	cv::circle(epipolar_img, points2[i], 3, color);
	}

	//output
	cvtColor(img_1, img_1rgb, COLOR_GRAY2BGR);
	hconcat(img_1rgb, epipolar_img, epipolar_img);
    cv::imwrite("LL-FAST-epipolar-5pt.png", epipolar_img);

    std::cout << "[DEBUG] Finished LL FAST" << std::endl;

  	return 0;
}