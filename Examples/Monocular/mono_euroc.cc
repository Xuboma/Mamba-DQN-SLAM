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
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<opencv2/core/core.hpp>
#include "ImuTypes.h"
#include<System.h>
#include<time.h>


using namespace std;
// clock_t start,finsh;
time_t first, second;
void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);
// extern bool flag_1;
int main(int argc, char **argv)
{
    // remove("/home/mxb/ORB_SLAM3/DDPG-ORBSLAM3/ORB_SLAM3-master/kf_dataset-MH01_mono.txt");
    // remove("/home/mxb/ORB_SLAM3/DDPG-ORBSLAM3/ORB_SLAM3-master/f_dataset-MH01_mono.txt");
    remove("/home/wwp/mxb/ORB_SLAM3-master/kf_dataset-MH01_mono.txt");
    remove("/home/wwp/mxb/ORB_SLAM3-master/f_dataset-MH01_mono.txt");
    remove("/home/wwp/mxb/ORB_SLAM3-master/read.txt");
    remove("/home/wwp/mxb/ORB_SLAM3-master/read_all.txt");
    remove("/home/wwp/mxb/ORB_SLAM3-master/result.txt");
    remove("/home/wwp/mxb/ORB_SLAM3-master/result_all.txt");
    if(argc < 5)
    {
        cerr << endl << "Usage: ./mono_euroc path_to_vocabulary path_to_settings path_to_sequence_folder_1 path_to_times_file_1 (path_to_image_folder_2 path_to_times_file_2 ... path_to_image_folder_N path_to_times_file_N) (trajectory_file_name)" << endl;
        return 1;
    }

    const int num_seq = (argc-3)/2;
    cout << "num_seq = " << num_seq << endl;
    bool bFileName= (((argc-3) % 2) == 1);
    string file_name;
    if (bFileName)
    {
        file_name = string(argv[argc-1]);
    }

    // Load all sequences: 加载所有图片序列
    int seq;
    vector< vector<string> > vstrImageFilenames;
    vector< vector<double> > vTimestampsCam;
    vector<int> nImages;
    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    nImages.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
//        cout << "Loading images for sequence " << seq << "...";
        //加载图片和曝光时间
        LoadImages(string(argv[(2*seq)+3]) + "/mav0/cam0/data", string(argv[(2*seq)+4]), vstrImageFilenames[seq], vTimestampsCam[seq]); 
//        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
    }
    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    //创建SLAM系统。 它初始化所有系统线程，并准备处理帧。
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR, true);

    int flag=0; //read.txt中标志id是否更改
    int index_flag=0;//read.txt中标志index值是否改变,如果不变则不跳出循环即一直用当前帧循环计算,如果改变则跳出循环,用下一帧进行计算
    double sF=1.2;
    int nL=8;
    float nartio = 0.9;
    bool one_time = true;
    bool flag_1=false;
    for (seq = 0; seq<num_seq; seq++)
    {
        // Main loop
        cv::Mat im;
        int proccIm = 0;
        // cout<<"vstrImageFilenames[0][0]:" << vstrImageFilenames[0][0] << endl;
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {
            // Read image from file
            im = cv::imread(vstrImageFilenames[seq][ni],CV_LOAD_IMAGE_UNCHANGED);  //不加改变的载入原图
            double tframe = vTimestampsCam[seq][ni];
            
            vector<ORB_SLAM3::IMU::Point> test;
            if(!flag_1)
            {
                ifstream infile; 
                int c;
                infile.open("switch.txt",ios::in);
                while (!infile.eof())
                    infile >> c;    //
                infile.close();   //关闭文件
                if(c==1)
                {
                    flag_1=true;
                    int r=0;
                    ofstream value("switch.txt",ios::ate);
                    value<<r<< endl;
                    value.close();
                }
                else
                {
                    flag_1=false;
                }
            }
            if(flag_1)
            {
                while(true)
                {
                    ifstream fTimes_read;
                    fTimes_read.open("read.txt");
                    string s_read;
                    getline(fTimes_read,s_read);
                    if(!s_read.empty()){
                        if(one_time){
                            // start = clock();
                            first=time(NULL);
                            one_time= false;
                        }
                        istringstream lineData(s_read);
                        int id;
                        int index;
                        double one;
                        int two;
                        float three;
                        lineData >>id>> index >> one >> two >> three;
                        if(flag<id){
                            sF=(double)one;
                            nL=two;
                            nartio = three;
                            flag=id;
                            if(im.empty())
                            {
                                cerr << endl << "Failed to load image at: "
                                    <<  vstrImageFilenames[seq][ni] << endl;
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
                            if (bFileName)
                            {
                                const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
                                const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
                                // const string t_file =  "tf_" + string(argv[argc-1]) + ".txt";
                                SLAM.SaveTrajectoryEuRoC(f_file);
                                SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
                                // SLAM.SaveTranslationEuRoC(t_file);
                                // cout<<"输出t_file文件"<<endl<<t_file<<endl;
                                        
                            }
                            else
                            {
                                SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
                                SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
                                // SLAM.SaveTranslationEuRoC("CameraTranslation.txt");
                            }
                            if(index_flag==0)
                            index_flag = index;

                            if(index_flag==index){
                                index_flag = index;
                            }else{
                                index_flag = index;
                                break;
                            }
                        }
                        else
                        {
                            usleep(1000);  //usleep以微妙为单位，sleep以秒为单位，1000000微秒us=1秒s
                        }
                    }
                    else
                    {
                        usleep(1000);
                    }

                    fTimes_read.close();

                }

            }
            else
            {//初始化进入即ni=0,1,2,3进入
                if(im.empty())
                {
                    cerr << endl << "Failed to load image at: "
                        <<  vstrImageFilenames[seq][ni] << endl;
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
                SLAM.SaveTrajectoryEuRoC(f_file);
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
    SLAM.Shutdown();
    return 0;
}

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);


    int i=0;
    // while(i<50)
    // {
    //     string s;
    //     getline(fTimes,s);
    //     if(!s.empty())
    //     {
    //         stringstream ss;
    //         ss << s;
    //         vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
    //         double t;
    //         ss >> t;
    //         vTimeStamps.push_back(t/1e9);

    //     }
    //     i=i+1;
    // }

    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);

        }
    }
}
