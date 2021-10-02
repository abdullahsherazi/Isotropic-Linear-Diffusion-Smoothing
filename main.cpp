#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// img ==> image to be diffused
// res_ilds ==> diffused image
// iter ==> diffusion time
// diffusivity ==> diffusivity parameter

void ilds(const Mat1b& src, Mat1b& dst, int iter, double diffusivity, double lambda = 0.25)
{
    Mat1f img;
    src.convertTo(img, CV_32F);
    lambda = fmax(0.001, std::fmin(lambda, 0.25)); // something in [0, 0.25] by default should be 0.25

    while (iter--)
    {
        Mat1f lap;
        Laplacian(img, lap, CV_32F);
        img += lambda * diffusivity * lap;
    }

    img.convertTo(dst, CV_8U);
}

// this class is to develop the data_set according to the questions
class Data {
    string image_name;
    double diffusivity;
    string question;
    int diffusion_time;
    
public:
    void set_data(string imagename, double diff, string question_no, int diff_time) {
        image_name = imagename;
        diffusivity = diff;
        question = question_no;
        diffusion_time = diff_time;
        }
    
    string get_image_name() { return image_name; }
    double get_diffusivity() { return diffusivity; }
    string get_question() { return question; }
    int get_diffusion_time() { return diffusion_time; }
};
int main() {
    
    //    this array according to the data provided in the questions
    Data data_array[9];
    data_array[0].set_data("office_noisy.png", 1, "Question_1_t=1_d=1", 1);
    data_array[1].set_data("office_noisy.png", 1, "Question_1_t=5_d=1", 5);
    data_array[2].set_data("office_noisy.png", 1, "Question_1_t=10_d=1", 10);
    data_array[3].set_data("office_noisy.png", 1, "Question_1_t=30_d=1", 30);
    data_array[4].set_data("office_noisy.png", 1, "Question_1_t=100_d=1", 100);
    data_array[5].set_data("office_noisy.png", 1, "Question_2_d=1_t=10", 10);
    data_array[6].set_data("office_noisy.png", 5, "Question_2_t=10_d=5", 10);
    data_array[7].set_data("office_noisy.png", 10, "Question_2_t=10_d=10", 10);
    data_array[8].set_data("office_noisy.png", 1, "Question_3_t=10_d=10", 30);

    //  looping through the data_array
    for (int i=0; i<sizeof(data_array); i++) {
        Mat1b res_ilds;
        
        // put your image folder path in path_image_folder
        string path_image_folder = "/Users/sami/Desktop/Canada/Assignment/";
        Mat1b img = imread(path_image_folder+data_array[i].get_image_name(), IMREAD_GRAYSCALE);
      
        ilds(img, res_ilds, data_array[i].get_diffusion_time() , data_array[i].get_diffusivity());
        // it will show diffused image according to the question.
        // after running code keep pressing enter to show new diffused image accroding to the data
        // provided through looping.
        imshow(data_array[i].get_question(), res_ilds);
        waitKey();
    }
    return 0;
}
