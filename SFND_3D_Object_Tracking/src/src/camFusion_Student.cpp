
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // SELECT Those keypoint that is in the boundingbox
    vector<cv::DMatch> kptsbounding;
    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        //take the pt in the curr
        cv::KeyPoint kp = kptsCurr.at(it->trainIdx);
        if(boundingBox.roi.contains(cv::Point(kp.pt.x, kp.pt.y)))
        {
            kptsbounding.push_back(*it);
        }
    }
    double dist_mean = 0;
    for (auto it = kptsbounding.begin(); it != kptsbounding.end(); ++it)
    {
        dist_mean += it->distance;
    }
    cout << "Find " << kptsbounding.size()  << " matches" << endl;
    if (kptsbounding.size() > 0)
    {
        dist_mean = dist_mean/kptsbounding.size();
    }
    else
    {
        return;
    }
    
    double threshold = dist_mean * 0.7;
    //select those points whose distance is smaller than the threshold
    for (auto it = kptsbounding.begin();it != kptsbounding.end(); ++it)
    {
        if(it->distance < threshold)
        {
            boundingBox.kptMatches.push_back(*it);
        }
    }
    cout << "Leave " << boundingBox.kptMatches.size()  << " matches" << endl;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // for loop for all the kptMatches
    vector<double> distRatios; //stores all the distratios
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1)
    {
        //get the outer keypoint
        cv::KeyPoint kptOuterPrev = kptsPrev.at(it1->queryIdx);
        cv::KeyPoint kptOuterCurr = kptsCurr.at(it1->trainIdx);
        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {//inner kpt -loop
            cv::KeyPoint kptInnerPrev = kptsPrev.at(it2->queryIdx);
            cv::KeyPoint kptInnerCurr = kptsCurr.at(it2->trainIdx);

            double minDist = 100.0;//min.required distance

            //compute the distance between the inner point and the outer point
            double distPrev = cv::norm(kptOuterPrev.pt - kptInnerPrev.pt);
            double distCurr = cv::norm(kptOuterCurr.pt - kptInnerCurr.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }//end of the inner loop
    }//end of the outer loop

    //only continue if list of distratip is not zero
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    //use the mid distance ratio
    std::sort(distRatios.begin(),distRatios.end());
    long medIndex = floor(distRatios.size()/2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex])/2.0 :distRatios[medIndex];

    //calculate the TTC
    double dT = 1/ frameRate;
    TTC = -dT / (1 - medDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    //find the closest the point of lidatpoint x but try to be robust
    double dT = 1/frameRate;
    double lanewidth = 4.0;
    vector<double> x_prev,x_cur;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end();++it)
    {
        if (abs(it->y) < lanewidth/2)
        {
            x_prev.push_back(it->x);
        }
    }
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end();++it)
    {
        if (abs(it->y) < lanewidth/2)
        {
            x_cur.push_back(it->x);
        }
    }
    //calculate the average time of the x_prev and x_cur
    double minxprev,minxcur;
    for (auto it = x_prev.begin(); it != x_prev.end(); ++it)
    {
        minxprev += *it;
    }
    minxprev = minxprev/x_prev.size();
    for (auto it = x_cur.begin(); it != x_cur.end(); ++it)
    {
        minxcur += *it;
    }
    minxcur = minxcur/x_cur.size();

    //calculate TTC
    cout << "minXCurr: " << minxcur << endl;
    cout << "minXPrev: " << minxprev << endl;
    TTC = minxcur * dT / (minxprev-minxcur);
}


// void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
// {
//     //find the bounding box in the query(prev) pic that has the most same keypoints(match) in the second pic
//     //cv::DMatch(int _queryIdx, int _trainIdx, float _distance)
//     int p = prevFrame.boundingBoxes.size();
//     int c = currFrame.boundingBoxes.size();
//     int count[p][c] = {};
//     //loop all the match and fill the count[p][c]
//     for (int i = 0; i < matches.size();++i)
//     {
//         cv::DMatch match = matches[i];
//         int queryidx = match.queryIdx;
//         int trainidx = match.trainIdx;
//         std::vector<int> prev_id, cur_id;
//         bool query_found = false;
//         bool train_found = false;
//         //see the keypoint of queryidx is in how many bounding_box of the prevframe
//         cv::KeyPoint prevkeypoint = prevFrame.keypoints[queryidx];
//         auto query_pt = cv::Point(prevkeypoint.pt.x, prevkeypoint.pt.y);
//         for (int j = 0; j < p; j++)
//         {
            
//             if(prevFrame.boundingBoxes[j].roi.contains(query_pt))
//             {
//                 query_found = true;
//                 prev_id.push_back(j);
//             }
//         }
//         //see the keypoint of trainidx is in how many bounding_box of the currFrame
//         cv::KeyPoint curkeypoint = currFrame.keypoints[trainidx];
//         auto train_pt = cv::Point(curkeypoint.pt.x, curkeypoint.pt.y);
//         for (int j = 0; j < c; j++)
//         {
            
//             if(prevFrame.boundingBoxes[j].roi.contains(train_pt))
//             {
//                 train_found = true;
//                 cur_id.push_back(j);
//             }
//         }
//         //if we found the point in the query set and the training set, add it into the count
//         if (train_found && query_found)
//         {
//             for (auto id_prev: prev_id)
//                 for (auto id_curr: cur_id)
//                      count[id_prev][id_curr] += 1;
//         }

//     }
//     // add the idx that has the most points in the second pic.
//     for (int i = 0; i < p; i++)
//     {
//         int max_count = 0;
//         int max_idx = 0;
//         for (int j = 0; j < c; j++)
//         {
//             if(count[i][j] > max_count)
//             {
//                 max_count = count[i][j];
//                 max_idx = j;
//             }
//         }
//         bbBestMatches[i] = max_idx;
//     }
//     bool bMsg = true;
//     if (bMsg)
//         for (int i = 0; i < p; i++)
//              cout << "Box " << i << " matches " << bbBestMatches[i]<< " box" << endl;
// }

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
    int p = prevFrame.boundingBoxes.size();
    int c = currFrame.boundingBoxes.size();
    int pt_counts[p][c] = { };
    for (auto it = matches.begin(); it != matches.end() - 1; ++it)     {
        cv::KeyPoint query = prevFrame.keypoints[it->queryIdx];
        auto query_pt = cv::Point(query.pt.x, query.pt.y);
        bool query_found = false;
        cv::KeyPoint train = currFrame.keypoints[it->trainIdx];
        auto train_pt = cv::Point(train.pt.x, train.pt.y);
        bool train_found = false;
        std::vector<int> query_id, train_id;
        for (int i = 0; i < p; i++) {
            if (prevFrame.boundingBoxes[i].roi.contains(query_pt))             {
                query_found = true;
                query_id.push_back(i);
             }
        }
        for (int i = 0; i < c; i++) {
            if (currFrame.boundingBoxes[i].roi.contains(train_pt))             {
                train_found= true;
                train_id.push_back(i);
            }
        }
        if (query_found && train_found)
        {
            for (auto id_prev: query_id)
                for (auto id_curr: train_id)
                     pt_counts[id_prev][id_curr] += 1;
        }
    }

    for (int i = 0; i < p; i++)
    {
         int max_count = 0;
         int id_max = 0;
         for (int j = 0; j < c; j++)
             if (pt_counts[i][j] > max_count)
             {
                  max_count = pt_counts[i][j];
                  id_max = j;
             }
          bbBestMatches[i] = id_max;
    }
    bool bMsg = true;
    if (bMsg)
        for (int i = 0; i < p; i++)
             cout << "Box " << i << " matches " << bbBestMatches[i]<< " box" << endl;
}
