//       Matteo Spadetto [214352] ComputerVision Assing2        //
//  _______  _______  _______  ___   _______  __    _  _______  //
// |   _   ||       ||       ||   | |       ||  |  | ||       | //
// |  |_|  ||  _____||  _____||   | |    ___||   |_| ||____   | //
// |       || |_____ | |_____ |   | |   | __ |       | ____|  | //
// |       ||_____  ||_____  ||   | |   ||  ||  _    || ______| //
// |   _   | _____| | _____| ||   | |   |_| || | |   || |_____  //
// |__| |__||_______||_______||___| |_______||_|  |__||_______| //
//                                                              //

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>

/// Defining parameters ///
#define TO_GRAY 0          // RGB to GRAY conversion
#define TO_HSV 1           // RGB to HSV conversion
#define TO_YUV 2           // RGB to YUV conversion
#define BLUR_X 1           // Blur parameter on X
#define BLUR_Y 1           // Blur parameter on Y
#define HISTORY 1000       // MOG2 history
#define MIXT 200           // MOG2 number of mixtures
#define BG_RATIO 0.1       // MOG2 background ratio
#define LEARNING_RATE -1   // MOG2 learning rate
#define SIGMA 1            // MOG2 sigma
#define MORPH_ERD 0        // Erode mode
#define MORPH_DIL 1        // Dilate mode
#define ERD1_X 2           // First erosion parameter on X
#define ERD1_Y 3           // First erosion parameter on Y
#define DIL1_X 0           // First dilation parameter on X
#define DIL1_Y 6           // First dilation parameter on Y
#define THRESH 50          // Threshold for contours filling
#define ERD2_X 1           // Second erosion parameter on X
#define ERD2_Y 1           // Second erosion parameter on Y
#define DIL2_X 0           // Second dilation parameter on X
#define DIL2_Y 4           // Second dilation parameter on Y
#define MAX_TRAJ_CENTER 20 // How many centers will be used to draw trajectories

using namespace cv;
using namespace std;

/// Structures needed for processing ///
typedef struct
{
    size_t id;        // Person id
    double centerX;   // Person center on X
    double centerY;   // Person center on Y
    double bb_left;   // Person bounding box left points (on X)
    double bb_top;    // Person bounding box top points (on Y)
    double bb_width;  // Person bounding box width
    double bb_height; // Person bounding box height
    double traj;      // Person trajectory
} info_t;             // Person information

typedef struct
{
    Mat img;               // Frame image
    size_t frame_id;       // Frame id
    vector<info_t> people; // Vector made by the collection of people detected
} frame_t;                 // Frames with their informations

/// Storing the .JPG files in a video made by a vector of frames ///
void store_input(vector<frame_t> &video, vector<frame_t> &video_input, vector<String> &filenames)
{
    cout << "\033[1;35m[Staring]\033[0m"
         << " Storing input frames..." << endl;
    for (size_t i = 0; i < filenames.size(); i++) // Cycling all frames
    {
        frame_t frame;                    // Tmp frame
        frame.img = imread(filenames[i]); // Reading the file path
        video.push_back(frame);           // Pushing back the frame in the video to merge at the end
        video_input.push_back(frame);     // Pushing back the frame in the video to compute
        cout << "\033[1;33m[Processing]\033[0m"
             << " Frame: " << i << " || Input file: " << filenames[i] << endl;
    }
}

/// Conveting video from RGB to another color scale ///
void rgb_conversion(vector<frame_t> &video_input, size_t color_mode)
{
    cout << "\033[1;32m[Computing]\033[0m"
         << " RGB to YUV conversion..." << endl;
    vector<frame_t> video_frame(video_input.size()); // Tmp video
    for (size_t i = 0; i < video_input.size(); i++)  // Cycling all frames
    {
        switch (color_mode)
        {
        case TO_GRAY:
            cv::cvtColor(video_input[i].img, video_frame[i].img, COLOR_RGB2GRAY); // RGB to GRAY frames conversion
            break;
        case TO_HSV:
            cv::cvtColor(video_input[i].img, video_frame[i].img, COLOR_RGB2HSV); // RGB to HSV frames conversion
            break;
        case TO_YUV:
            cv::cvtColor(video_input[i].img, video_frame[i].img, COLOR_RGB2YUV); // RGB to YUV frames conversion
            break;
        default:
            // No color conversion
            break;
        }
        video_input[i] = video_frame[i]; // Updating the input video
    }
}

/// Applying gaussian blur ///
void gauss_blur(vector<frame_t> &video_input, size_t mask_x, size_t mask_y)
{
    cout << "\033[1;32m[Computing]\033[0m"
         << " Gaussian blur..." << endl;
    vector<frame_t> video_frame(video_input.size()); // Tmp video
    for (size_t i = 0; i < video_input.size(); i++)  // Cycling all frames
    {
        GaussianBlur(video_input[i].img, video_frame[i].img, Size(mask_x, mask_y), 0); // Gaussian blur
        video_input[i] = video_frame[i];                                               // Updating the input video
    }
}

/// Applying MOG2 ///
void mog2(vector<frame_t> &video_input, size_t hist, size_t mixt, double bg_ratio, double lr, double sigma)
{
    cout << "\033[1;32m[Computing]\033[0m"
         << " MOG2..." << endl;
    Ptr<bgsegm::BackgroundSubtractorMOG> pMOG;               // bgsegm
    Ptr<BackgroundSubtractorMOG2> pMOG2;                     // pMOG2
    vector<frame_t> video_frame(video_input.size());         // Tmp video
    pMOG2 = createBackgroundSubtractorMOG2(hist, 16, false); // Setting history, varThreshold, shadows
    for (size_t i = 0; i < video_input.size(); i++)          // Cycling all frames
    {
        Mat bg;
        pMOG2->apply(video_input[i].img, video_frame[i].img, lr); // MOG2
        pMOG2->getBackgroundImage(bg);                            // Updating background
        video_input[i] = video_frame[i];                          // Updating the input video
    }
}

void morph(vector<frame_t> &video_input, size_t x, size_t y, size_t morph_mode)
{
    cout << "\033[1;32m[Computing]\033[0m"
         << " Erosion with {x, y} = {" << x << ", " << y << "}" << endl;
    vector<frame_t> video_frame(video_input.size());                                                 // Tmp video
    Mat element_erd = getStructuringElement(MORPH_ELLIPSE, Size(2 * x + 1, 2 * y + 1), Point(x, y)); // Setting erosion
    Mat element_dil = getStructuringElement(MORPH_ELLIPSE, Size(2 * x + 1, 2 * y + 1), Point(x, y)); // Setting dilation
    for (size_t i = 0; i < video_input.size(); i++)
    {
        switch (morph_mode)
        {
        case MORPH_ERD:
            erode(video_input[i].img, video_frame[i].img, element_erd); // Eroding
            break;
        case MORPH_DIL:
            dilate(video_input[i].img, video_frame[i].img, element_dil); // Dilating
            break;
        default:
            break;
        }
        video_input[i] = video_frame[i];
    }
}

/// Drawing and filling the countours ///
void fill_contours(vector<frame_t> &video_input, size_t threshold)
{
    cout << "\033[1;32m[Computing]\033[0m"
         << " Contours erosion and dilation..." << endl;
    vector<frame_t> video_frame(video_input.size()); // Tmp video
    for (size_t i = 0; i < video_input.size(); i++)  // Cycling all frames
    {
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        Canny(video_input[i].img, video_frame[i].img, threshold, threshold * 3, 3);                             // Canny
        findContours(video_frame[i].img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0)); // Finding contours
        vector<vector<Point>> polygon(contours.size());
        vector<vector<Point>> hull(contours.size());
        for (size_t k = 0; k < contours.size(); k++) // Cycling all countours (that sould be a person)
        {
            approxPolyDP(contours[k], polygon[k], 2, true); // Approximation of the contours
            convexHull(polygon[k], hull[k], false, true);   // Repairing convess hull
            if (cv::arcLength(hull[k], true) >= 80)         // Only for contours whit a certain perimeter
            {
                fillConvexPoly(video_frame[i].img, hull[k], Scalar(255, 255, 255), 8, 0); // Fill iside the contours
            }
        }
        video_input[i] = video_frame[i]; // Updating the input video
    }
}

/// Drawing final contours ///
vector<frame_t> draw_contours(vector<frame_t> &video_input, size_t threshold)
{
    cout << "\033[1;32m[Computing]\033[0m"
         << " Contours drawing..." << endl;
    vector<frame_t> video_frame(video_input.size()); // Tmp video
    size_t id = 0;                                   // Initializing id as 1
    size_t id_tmp = 0;
    size_t frame_number = 1;                        // Initializing the frame_id as 1
    for (size_t i = 0; i < video_input.size(); i++) // Cycling all frames
    {
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        Canny(video_input[i].img, video_frame[i].img, threshold, threshold * 3, 3);                             // Canny
        findContours(video_frame[i].img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0)); // Finding contours
        vector<vector<Point>> polygon(contours.size());
        vector<vector<Point>> hull(contours.size());
        vector<Point2f> center(contours.size());
        vector<Rect> boundRect(contours.size());
        for (size_t k = 0; k < contours.size(); k++) // For each person
        {
            approxPolyDP(contours[k], polygon[k], 2, true); // Approximation of the contours
            convexHull(polygon[k], hull[k], false, true);   // Repairing convess hull
            if (cv::arcLength(hull[k], 1) >= 80)            // Only if the hull has a certain perimeter
            {
                Moments mom = moments(hull[k]);     // Initialize mom
                double centerX = mom.m10 / mom.m00; // Calaculating centerX
                double centerY = mom.m01 / mom.m00; // Calaculating centerY
                info_t person;                      // Tmp person
                if (centerX >= 20 && centerY >= 20)
                {
                    person.centerX = centerX;                                                                    // Person centerX
                    person.centerY = centerY;                                                                    // Person centerY
                    circle(video_frame[i].img, Point(centerX, centerY), 2, Scalar(255, 255, 255), 3, 8, 0);      // Drawing a circle on centers
                    drawContours(video_frame[i].img, hull, k, Scalar(255, 0, 255), 2, 8, hierarchy, 0, Point()); // drawing the final contours
                    boundRect[k] = boundingRect(polygon[k]);
                    rectangle(video_frame[i].img, boundRect[k].tl(), boundRect[k].br(), Scalar(255, 255, 255), 2, 8); // Drawing rectangle boxes
                    person.bb_height = boundRect[k].height;
                    person.bb_width = boundRect[k].width;
                    person.bb_top = boundRect[k].tl().y;
                    person.bb_left = boundRect[k].tl().x;
                    person.traj = 0; // Trajectory to overcame occlusions
                    if (i > 11)      // The first 11 frames are not good for the id because are used as initial infos (it is just the 1.4% of the video)
                    {
                        double trajectoryX = (person.centerX - video_input[i - 3].people[k].centerX); // TrajectoryX
                        double trajectoryY = (person.centerY - video_input[i - 3].people[k].centerY); // trajectoryY
                        double trajectory = trajectoryY / trajectoryX;                                // Direction = dY/dX
                        person.traj = trajectory;
                        size_t flag = 0;
                        for (size_t h = 1; h < 10; h++) // Cycling 10 frames before the current one
                        {
                            double distX_tmp = 1000;                                      // Tmp distance on X of two contours of different frames
                            double distY_tmp = 1000;                                      // Tmp distance on X of two contours of different frames
                            for (size_t g = 0; g < video_input[i - h].people.size(); g++) // Cycling all the contours of the frame
                            {
                                // If CentersX or CentersY of people (of two different frames) are near and the dimension of the
                                // rectangle bounding boxes are almost the same and the directions are almost the same
                                if (abs(person.bb_height - video_input[i - h].people[g].bb_height) <= 35 && abs(person.bb_width - video_input[i - h].people[g].bb_width) <= 35 && abs(person.centerX - video_input[i - h].people[g].centerX) <= 50 && abs(person.centerY - video_input[i - h].people[g].centerY) <= 50 && abs(trajectory) - abs(video_input[i - h].people[g].traj) <= 50)
                                {
                                    // If this contours is the nearest to the current contour
                                    if (abs(person.centerX - video_input[i - h].people[g].centerX) < distX_tmp && abs(person.centerY - video_input[i - h].people[g].centerY) < distY_tmp)
                                    {
                                        id_tmp = video_input[i - h].people[g].id;                               // Maintain the old id
                                        distX_tmp = abs(person.centerX - video_input[i - h].people[g].centerX); // Updating distX_tmp
                                        distY_tmp = abs(person.centerY - video_input[i - h].people[g].centerY); // Updating distY_tmp
                                        flag = 1;
                                    }
                                }
                            }
                        }
                        if (flag == 0) // If the conditions are not satisfied then there is a new person
                        {
                            id++;        // Generating a new id
                            id_tmp = id; // Store the new id in id_tmp
                        }
                    }
                    else // Just for the first 11 frames
                    {
                        id++;        // Generating a new id
                        id_tmp = id; // Storing the id in id_tmp
                    }

                    person.id = id_tmp;                      // Storing the id
                    video_input[i].people.push_back(person); // Inserting in input video
                }
            }
        }
        video_input[i].frame_id = frame_number;  // Storing the frame id
        frame_number++;                          // Updating the frame id
        video_input[i].img = video_frame[i].img; // Storing the image
    }
    return video_input;
}

/// Merging videos for the output ///
void merge_frame(vector<frame_t> &video1, vector<frame_t> &video2, vector<frame_t> &video_output)
{
    for (size_t i = 0; i < video1.size(); i++)
    {
        frame_t frame;
        cvtColor(video1[i].img, video1[i].img, COLOR_GRAY2RGB); // Metching the color scales
        frame.img = video2[i].img + video1[i].img;              // Merging the two videos
        video_output.push_back(frame);
    }
}

/// Showing and saving the videos ///
void save_show_video(vector<frame_t> &video1, String s1, vector<frame_t> &video2, String s2, vector<frame_t> &video)
{
    cout << "\033[1;36m[Showing]\033[0m"
         << " Showing videos..." << endl;

    std::ofstream detection_file("./detection.txt");                        // Initializing file path for the detection parameters
    std::ofstream tracking_file("./tracking.txt");                          // Initializing file path for the tracking parameters
    VideoWriter outputVideo;                                                // Initializing output video
    outputVideo.open("./video_out.avi", 0, 15, video2[0].img.size(), true); // Initializing file path and code for the output video
    for (size_t i = 0; i < video1.size(); i++)                              // Cycling all the frames
    {
        for (size_t j = 0; j < video[i].people.size(); j++) // Cycling all people
        {
            // Printing all the parameters of each person
            cout << "\033[1;36m[Info] Frame: \033[0m" << video[i].frame_id << "\033[1;36m | id: \033[0m" << video[i].people[j].id
                 << "\033[1;36m | centerX: \033[0m" << video[i].people[j].centerX << "\033[1;36m | centerY: \033[0m" << video[i].people[j].centerY
                 << "\033[1;36m | bb_left: \033[0m" << video[i].people[j].bb_left << "\033[1;36m | bb_top: \033[0m" << video[i].people[j].bb_top
                 << "\033[1;36m | bb_width: \033[0m" << video[i].people[j].bb_width << "\033[1;36m | bb_height: \033[0m" << video[i].people[j].bb_height << endl;
            // Saving all parameters needed for "detection.txt" and "tracking.txt" files
            detection_file << video[i].frame_id << "," << video[i].people[j].bb_left << "," << video[i].people[j].bb_top << "," << video[i].people[j].bb_width << "," << video[i].people[j].bb_height << std::endl;
            tracking_file << video[i].frame_id << "," << video[i].people[j].id << "," << video[i].people[j].centerX << "," << video[i].people[j].centerY << std::endl;
        }
        imshow(s1, video1[i].img); // Showing first video
        {
            for (size_t m = 0; m < video[i].people.size(); m++) // Cycling each person
            {
                size_t traj_centers = MAX_TRAJ_CENTER;
                if (i < traj_centers) // If they are first frames
                {
                    traj_centers = i; // All history
                }
                else
                {
                    traj_centers = MAX_TRAJ_CENTER; // Only the MAX_TRAJ_CENTER frames
                }
                for (size_t n = 0; n < traj_centers; n++) // Cycling to draw the center points that make trajextories
                {
                    // Just to avoid centers on top-left corner of the frame. It is an opencv problem. The files are correct
                    if (video[i - n].people[m].centerX >= 10 && video[i - n].people[m].centerY >= 10)
                    {
                        Scalar color = Scalar(0, 100, 240);
                        circle(video2[i].img, Point(video[i - n].people[m].centerX, video[i - n].people[m].centerY), 2, color, 3, 8, 0); // Drwaing trajectories
                    }
                }
            }
        }
        imshow(s2, video2[i].img);        // Showing second video
        outputVideo.write(video2[i].img); // Writing all datas in the output video
        waitKey(80);                      // Waiting for marge the FPS
    }
    detection_file.close(); // Closing "detection.txt"
    tracking_file.close();  // Closing "tracking.txt"
    outputVideo.release();  // Closing the output video
}

int main(int argc, char const *argv[])
{
    vector<String> filenames;     // Path
    vector<frame_t> video_input;  // Input video
    vector<frame_t> video;        // Tmp video
    vector<frame_t> video_output; // Output video

    String folderpath = "./Video/img1/";                                              // First part of the path
    cv::glob(folderpath, filenames);                                                  // Path
    store_input(video, video_input, filenames);                                       // Storing the input frames in a video
    rgb_conversion(video, TO_YUV);                                                    // Conveting from RGB to YUV
    gauss_blur(video, BLUR_X, BLUR_Y);                                                // Gaussian blur
    mog2(video, HISTORY, MIXT, BG_RATIO, LEARNING_RATE, SIGMA);                       // Using MOG2
    morph(video, ERD1_X, ERD1_Y, MORPH_ERD);                                          // Erosion
    morph(video, DIL1_X, DIL1_Y, MORPH_DIL);                                          // Dilation
    fill_contours(video, THRESH);                                                     // Filling contours
    morph(video, ERD2_X, ERD2_Y, MORPH_ERD);                                          // Erosion
    morph(video, DIL2_X, DIL2_Y, MORPH_DIL);                                          // Dilation
    video = draw_contours(video, THRESH);                                             // Drawing final contours
    merge_frame(video, video_input, video_output);                                    // Merging the videos
    save_show_video(video_input, "Video input", video_output, "Video output", video); // Showing and saving all datas
    cout << "\033[1;31m[COMPLETED WITHOUT ERRORS]\033[0m" << endl;

    return 0;
}