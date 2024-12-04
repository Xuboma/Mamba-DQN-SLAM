/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<System.h>
#include<time.h>
#include<algorithm>
#include<vector>
#include<fstream>
#include<chrono>
#include<iomanip>
#include <unistd.h>

#include<opencv2/core/core.hpp>

#include"System.h"
#include "Converter.h"

using namespace std;
time_t first, second;

//1.LoadImages读取图片路径和时间戳
void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

std::string extract_filename(const std::string& path) {
    // 找到最后一个斜杠的位置
    size_t last_slash_pos = path.find_last_of('/');
    if (last_slash_pos == std::string::npos) {
        return path; // 如果没有找到斜杠，返回整个字符串
    }

    // 提取图片名
    std::string filename = path.substr(last_slash_pos + 1);

    // 找到最后一个点的位置
    size_t last_dot_pos = filename.find_last_of('.');
    if (last_dot_pos != std::string::npos) {
        // 去掉文件扩展名
        filename = filename.substr(0, last_dot_pos);
    }

    return filename;
}

double ttrack_tot = 0;
int main(int argc, char **argv)
{
    //删除文件
    // remove("/home/hx/ORB_SLAM3/kf_dataset-room1_512_mono.txt");
    // remove("/home/hx/ORB_SLAM3/f_dataset-room1_512_mono.txt");
    //remove("/home/hx/ORB_SLAM3/read.txt");
    //remove("/home/hx/ORB_SLAM3/read_all.txt");
    //remove("/home/hx/ORB_SLAM3/result.txt");
    //remove("/home/hx/ORB_SLAM3/result_all.txt");



    const int num_seq = (argc-3)/2;
    cout << "num_seq = " << num_seq << endl;
    bool bFileName= (((argc-3) % 2) == 1);

    string file_name;
    if (bFileName)
    {
        file_name = string(argv[argc-1]);
        cout << "file name: " << file_name << endl;
    }

    //检查参数是否足够
    if(argc < 4)
    {
        cerr << endl << "Usage: ./mono_tum_vi path_to_vocabulary path_to_settings path_to_image_folder_1 path_to_times_file_1 (path_to_image_folder_2 path_to_times_file_2 ... path_to_image_folder_N path_to_times_file_N) (trajectory_file_name)" << endl;
        return 1;
    }

    // Load all sequences:加载所有图片序列
    int seq;

    //加载图像，图像序列的文件名字符串序列
    vector< vector<string> > vstrImageFilenames;
    vector< vector<double> > vTimestampsCam;
    vector<int> nImages;

    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    nImages.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";
        //加载图片和曝光时间

        //这里和mono_euroc.cc的内容不太一样
        LoadImages(string(argv[(2*seq)+3]), string(argv[(2*seq)+4]), vstrImageFilenames[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;


        //当前图像序列中，图像数目
        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];

        if((nImages[seq]<=0))
        {
            cerr << "ERROR: Failed to load images for sequence" << seq << endl;
            return 1;
        }

    }
    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.

    //step2  加载SLAM系统
    //创建 SLAM系统，初始化所有线程，并准备处理帧
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR,true);


    //step3   运行前的准备

    //添加的部分1，开始

    int flag = 0;
    int index_flag = 0;
    double sF = 1.2;
    int nL = 8;
    float nartio = 0.9;
    bool one_time = true;
    bool flag_1 = false;

    //到这里结束

    int proccIm = 0;

    //step4 依次跟踪序列中的每一张图片
    for (seq = 0; seq<num_seq; seq++)
    {

        // Main loop
        cv::Mat im;
        proccIm = 0;

        //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {

            //step4.1 根据前面的文件名，读取图像，读取过程中不改变图像的格式
            // Read image from file
            im = cv::imread(vstrImageFilenames[seq][ni],cv::IMREAD_GRAYSCALE);
            string img_name=extract_filename(vstrImageFilenames[seq][ni]);
            cout<<img_name<<endl;
            //im = cv::imread(vstrImageFilenames[seq][ni],CV_LOAD_IMAGE_UNCHANGED);  //不加改变的载入原图

            // clahe
            //clahe->apply(im,im);

            //读取时间戳
            // cout << "mat type: " << im.type() << endl;
            double tframe = vTimestampsCam[seq][ni];


            //添加的第二部分，开始
            vector<ORB_SLAM3::IMU::Point> test;
            if(!flag_1){
                ifstream infile;
                int c;
                infile.open("switch.txt", ios::in);
                while(!infile.eof())
                    infile >> c;
                infile.close();
                if(c==1){
                    flag_1 = true;
                    int r = 0;
                    ofstream value("switch.txt", ios::ate);
                    value << r << endl;
                    value.close();
                }
                else{
                    flag_1 = false;
                }
            }
            if(flag_1)
            {
                while(true)
                {
                    ifstream fTimes_read;
                    fTimes_read.open("read.txt");
                    string s_read;
                    getline(fTimes_read, s_read);
                    if(!s_read.empty()){
                        if(one_time){
                            first = time(NULL);
                            one_time = false;
                        }
                        
                        istringstream lineData(s_read);
                        int id;
                        int index;
                        double one;
                        int two;
                        float three;
                        lineData >> id >> index >> one >> two >> three;

                        if(flag<id)
                        {
                            sF = (double)one;
                            nL = two;
                            nartio = three;
                            flag = id;
                            //添加第二部分到这里结束


                            //step4.2 检查图像合法性
                            if(im.empty())
                            {
                                cerr << endl << "Failed to load image at: "
                                <<  vstrImageFilenames[seq][ni] << endl;
                                return 1;
                            }

                            //step4.3 开始计时 
                            #ifdef COMPILEDWITHC11
                                std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                            #else
                                std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
                            #endif

                                //添加的部分3，开始
                                test.push_back(ORB_SLAM3::IMU::Point(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0));
                                //到这里结束

                                //step4.4 追踪当前图像
                                //TrackMonocular将图片传给slam系统
                                // Pass the image to the SLAM system
                                SLAM.TrackMonocular(im,tframe,test,"",sF,nL,nartio); // TODO change to monocular_inertial
                            
                            //step4.5 追踪完成，停止当前帧图像计时，并计算耗时
                            #ifdef COMPILEDWITHC11
                                std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
                            #else
                                std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
                            #endif

                            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
                            ttrack_tot += ttrack;

                            vTimesTrack[ni]=ttrack;

                            // Wait to load the next frame
                            //step4.6 
                            //根据图像时间戳中记录的两张图像之间的时间和现在追踪当前图像所耗费的时间，
                            //继续等待指定的时间以使得下一张图像能够按照时间戳被送入到SLAM系统中进行跟踪
                            double T=0;
                            if(ni<nImages[seq]-1)
                                T = vTimestampsCam[seq][ni+1]-tframe;
                            else if(ni>0)
                                T = tframe-vTimestampsCam[seq][ni-1];

                            if(ttrack<T)
                                usleep((T-ttrack)*1e6); // 1e6   
                            
                            // Tracking time statistics

                            // Save camera trajectory

                            if (bFileName)
                            {
                                const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
                                const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
        
                                SLAM.SaveTrajectoryEuRoC(f_file,img_name);
                                SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
                            }
                            else
                            {
                                SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt",img_name);
                                SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
                            }

                            sort(vTimesTrack.begin(),vTimesTrack.end());
                            float totaltime = 0;
                            for(int ni=0; ni<nImages[0]; ni++)
                            {
                                totaltime+=vTimesTrack[ni];
                            }
                            cout << "-------" << endl << endl;
                            cout << "median tracking time: " << vTimesTrack[nImages[0]/2] << endl;
                            cout << "mean tracking time: " << totaltime/proccIm << endl;

                            if(index_flag==0)
                                index_flag = index;
                            if(index_flag==index){
                                index_flag = index;
                            }else{
                                index_flag = index;
                                break;
                            }
                        }else{
                            usleep(1000);
                        }
                    }else{
                        usleep(1000);
                    }
                    fTimes_read.close();
                }
            }

            else{
                //初始化进入
                if(im.empty()){
                    cerr << endl
                         << "Failed to load image at: " << vstrImageFilenames[seq][ni] << endl;
                    return 1;
                }

                #ifdef COMPILEDWITHC11
                    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                #else
                    std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
                #endif
                    test.push_back(ORB_SLAM3::IMU::Point(1.0,1.0,1.0,1.0,1.0,1.0,1.0));
                    // Pass the image to the SLAM system
                    // cout << "tframe = " << tframe << endl;
                    SLAM.TrackMonocular(im,tframe,test,"",sF,nL,nartio); // TODO change to monocular_inertial

                #ifdef COMPILEDWITHC11
                    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
                #else
                    std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
                #endif

                double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

                vTimesTrack[ni]=ttrack;

                // Wait to load the next frame
                double T=0;
                if(ni<nImages[seq]-1)
                    T = vTimestampsCam[seq][ni+1]-tframe;
                else if(ni>0)
                    T = tframe-vTimestampsCam[seq][ni-1];

                if(ttrack<T)
                    usleep((T-ttrack)*1e6); // 1e6
                const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
                SLAM.SaveTrajectoryEuRoC(f_file,img_name);
            }
    
        }
        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;
            SLAM.ChangeDataset();
        }

    }
    second=time(NULL); 
    cout<<"------------ first -----------"<<first<<endl;
    cout<<"------------ second -----------"<<second<<endl;
    printf("time is: %f seconds",difftime(second,first));
    printf("\n");
    // Stop all threads

    // step5  中止当前slam系统

    // cout << "ttrack_tot = " << ttrack_tot << std::endl;
    // Stop all threads
    SLAM.Shutdown();

    return 0;
}

//从文件中加载图像序列，加载每一张图像的文件路径和时间戳
void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    //打开文件
    ifstream fTimes;
    cout << strImagePath << endl;
    cout << strPathTimes << endl;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);

        //只有当前行不为空时执行
        if(!s.empty())
        {
            stringstream ss;
            ss << s;

            //生成当前行所指出RGB图像的文件名称
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            //记录图像的时间戳
            vTimeStamps.push_back(t/1e9);

        }
    }
}


