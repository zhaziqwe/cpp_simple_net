#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>
//基于https://www.bilibili.com/video/BV1Y64y1z7jM?p=2&vd_source=fab2d91e511474675f13167b291180ef
//参数基本没变，训练一个类似异或函数，相同为0，不同为1
//定义每层节点数
#define INNODE 2
#define HIDENODE 4
#define OUTNODE 1

//定义超参数
double rate = 0.8;
double threshold = 1e-4;
size_t mosttimes = 1e6;

//样本结构，输入和输出
struct Sample {
    std::vector<double> in,out;
};

//节点结构，存计算的值，偏置值，偏置值求导，权值，权值求导，由于全连接，w为向量
struct Node{
    double value{}, bias{}, bias_delta{};
    std::vector<double> weight,weight_delta;
};


//工具空间，定义一些方便函数
namespace  utils {
    //sigmoid激活函数
    inline double sigmoid(double x) {
        double res = 1.0 / (1.0 + std::exp(-x));
        return res;
    }

    //读取文件
    std::vector<double> getFileData(std::string filename){
        std::vector<double> res;
        std:: ifstream in(filename);
        if(in.is_open())
        {
            while(!in.eof()){
                double buffer;
                in >> buffer;
                res.push_back((buffer));
            }
            in.close();
        }
        else
        {
            std::cout<<"Error in reading "<<filename<<std::endl;
        }
        return res;
    }

    //读取训练样本
    std::vector<Sample> getTrainData(std::string filename){
        std::vector<Sample> res;

        std::vector<double> buffer = getFileData(filename);

        for(size_t i=0;i<buffer.size();i+=INNODE+OUTNODE){
            Sample tmp;
            for(size_t t=0;t<INNODE;t++){
                tmp.in.push_back(buffer[i+t]);
            }
            for(size_t t=0;t<OUTNODE;t++) {
                tmp.out.push_back(buffer[i+INNODE+t]);
            }
            res.push_back(tmp);
        }
        return res;
    }

    //读取测试样本
    std::vector<Sample> getTestData(std::string filename){
        std::vector<Sample> res;

        std::vector<double> buffer = getFileData(filename);

        for(size_t i=0;i<buffer.size();i+=INNODE+OUTNODE){
            Sample tmp;
            for(size_t t=0;t<INNODE;t++){
                tmp.in.push_back(buffer[i+t]);
            }
            res.push_back(tmp);
        }
        return res;
    }

}

//定义（数组）输入层，隐藏层，输出层，存节点
Node *inputLayer[INNODE],*hideLayer[HIDENODE],*outLayer[OUTNODE];

//初始化所有的节点，全部层都要遍历，随机初始参数，求导初始为0
inline void init() {
    std::mt19937 rd;
    rd.seed(std::random_device()());

    std::uniform_real_distribution<double> distribution(-1,1);
    for(size_t i=0;i<INNODE;i++){
        ::inputLayer[i] =  new Node();
        for(size_t j=0;j<HIDENODE;j++){
            ::inputLayer[i]->weight.push_back(distribution(rd));
            ::inputLayer[i]->weight_delta.push_back(0.f);
        }
    }

    for(size_t i=0; i< HIDENODE;i++){
        ::hideLayer[i] = new Node();
        ::hideLayer[i]->bias = distribution(rd);
        for(size_t j=0; j< OUTNODE; j++)
        {
            ::hideLayer[i]->weight.push_back(distribution(rd));
            ::hideLayer[i]->weight_delta.push_back(0.f);
        }
    }

    for(size_t i=0;i<OUTNODE;i++){
        ::outLayer[i] = new Node();
        ::outLayer[i]->bias = distribution(rd);
    }

}
//重置导数，因为每次都要累加，所以需要清0
inline void reset_delta(){
    for(size_t i=0; i<INNODE; i++){
        ::inputLayer[i]->weight_delta.assign(::inputLayer[i]->weight_delta.size(),0.f);
    }

    for(size_t i=0;i<HIDENODE;i++){
        ::hideLayer[i]->bias_delta = 0.f;
        ::hideLayer[i]->weight_delta.assign(::hideLayer[i]->weight_delta.size(),0.f);
    }

    for(size_t i=0;i<OUTNODE;i++){
        ::outLayer[i] ->bias_delta = 0.f;
    }

}
//前向计算和反向传播
int main(int argc, char *argv[]) {
    //初始化
    init();
    //读取样本
    std::vector<Sample> train_data = utils::getTrainData("traindata.txt");
    //开始迭代
    for(size_t times =0; times<mosttimes; times++){
        //重置导数
        reset_delta();
        //定义误差范围
        double error_max = 0.f;
        //对每个样本训练
        for(auto &idx:train_data){

            //输入层赋值
            for(size_t i=0;i<INNODE;i++){
                ::inputLayer[i]->value = idx.in[i];
            }

            //前向计算hideLayer层的值
            for(size_t j=0; j<HIDENODE;j++){
                double sum = 0;
                for(size_t i=0; i<INNODE;i++){
                    sum+=::inputLayer[i]->value * ::inputLayer[i]->weight[j];
                }
                sum -= ::hideLayer[j]->bias;

                ::hideLayer[j]->value = utils::sigmoid(sum);
            }

            //前向计算outLayer层的值
            for(size_t j=0;j<OUTNODE;j++){
                double sum=0;
                for(size_t i=0;i<HIDENODE;i++)
                {
                    sum+=::hideLayer[i]->value * ::hideLayer[i]->weight[j];
                }
                sum -= ::outLayer[j]->bias;

                ::outLayer[j]->value = utils::sigmoid(sum);
            }

            //定义误差和损失函数
            double error = 0.f;
            for(size_t i=0;i<OUTNODE;i++){
                double tmp = std::fabs(::outLayer[i]->value - idx.out[i]);
                error += tmp*tmp/2;
            }

            error_max = std::max(error_max,error);

//            double attr_delta = 0.f;
//            for(size_t i=0;i<OUTNODE;i++){
//                attr_delta += (idx.out[i] - ::outLayer[i]->value) *
//                        ::outLayer[i]->value * (1.0- ::outLayer[i]->value);
//            }

            //反向传播 全是按手推的公式计算，没有复用性，需要重构
            //计算out层的bias_delta
            for(size_t i=0;i<OUTNODE;i++){
                double bias_data = -(idx.out[i] - ::outLayer[i]->value) *
                        ::outLayer[i]->value *
                        (1.0- ::outLayer[i]->value);
                ::outLayer[i]->bias_delta += bias_data;

            }

            //计算hide层的weight_delta和bias_delta
            for(size_t i=0;i<HIDENODE;i++){
                for(size_t j=0;j<OUTNODE;j++){
                    double weight_delta = (idx.out[j] - ::outLayer[j]->value) *
                            ::outLayer[j]->value * (1.0- ::outLayer[j]->value) *
                            ::hideLayer[i]->value;
                    ::hideLayer[i]->weight_delta[j]+=weight_delta;

                }
            }
            for(size_t i=0;i<HIDENODE;i++){
                double sum=0;
                for(size_t j=0;j<OUTNODE;j++){
                    sum+= -(idx.out[j]-::outLayer[j]->value) *
                            ::outLayer[j] ->value *
                            (1.0 - ::outLayer[j]->value) *
                            ::hideLayer[i]->weight[j];
                }
                ::hideLayer[i]->bias_delta+=
                        sum * ::hideLayer[i]->value *(1.0 - ::hideLayer[i]->value);
            }

            //计算in层的weight_delta和bias_delta
            for(size_t i=0; i<INNODE;i++){
                for(size_t j=0;j<HIDENODE;j++){
                    double sum=0.f;
                    for(size_t k=0;k<OUTNODE;k++){
                        sum+= (idx.out[k] - ::outLayer[k]->value) *
                                ::outLayer[k]->value *
                                (1.0-::outLayer[k]->value) *
                                ::hideLayer[j]->weight[k];

                    }
                    ::inputLayer[i]->weight_delta[j]+=
                            sum *
                            ::hideLayer[j]->value *
                            (1.0 - ::hideLayer[j]->value) *
                            ::inputLayer[i]->value;
                }
            }

        }

        //检测是否达到误差
        if(error_max<::threshold){
            std::cout<<"Success with"<< times+1<<"times training." <<std::endl;
            std::cout<<"Max error:"<<error_max<< std::endl;
            break;
        }

        //样本数
        auto train_data_size = double (train_data.size());

        //因为是累加所以需要取平均值来更新所有参数
        for(size_t i=0; i<INNODE; i++){
            for(size_t j=0;j<HIDENODE;j++){
                ::inputLayer[i]->weight[j] +=
                        rate * ::inputLayer[i]->weight_delta[j]/train_data_size;
            }
        }

        for(size_t i=0; i<HIDENODE;i++){
            ::hideLayer[i]->bias +=
                    rate * ::hideLayer[i]->bias_delta/train_data_size;
            for(size_t j=0; j<OUTNODE;j++){
                ::hideLayer[i]->weight[j]+=
                        rate*::hideLayer[i]->weight_delta[j]/train_data_size;
            }
        }

        for(size_t i=0;i<OUTNODE;i++){
            ::outLayer[i]->bias+=
                    rate* ::outLayer[i]->bias_delta/train_data_size;
        }

    }

    //测试过程
    std::vector<Sample> test_data = utils::getTestData("testdata.txt");

    for(auto &idx: test_data){
        //输入层赋值
        for(size_t i=0;i<INNODE;i++){
            ::inputLayer[i]->value = idx.in[i];
        }

        //前向计算hideLayer层的值
        for(size_t j=0; j<HIDENODE;j++){
            double sum = 0;
            for(size_t i=0; i<INNODE;i++){
                sum+=::inputLayer[i]->value * ::inputLayer[i]->weight[j];
            }
            sum -= ::hideLayer[j]->bias;

            ::hideLayer[j]->value = utils::sigmoid(sum);
        }

        //前向计算outLayer层的值,并赋值,最后输出
        for(size_t j=0;j<OUTNODE;j++){
            double sum=0;
            for(size_t i=0;i<HIDENODE;i++)
            {
                sum+=::hideLayer[i]->value * ::hideLayer[i]->weight[j];
            }
            sum -= ::outLayer[j]->bias;

            ::outLayer[j]->value = utils::sigmoid(sum);
            idx.out.push_back(::outLayer[j]->value);

            for(auto &tmp:idx.in){
                std::cout<<tmp<<" ";
            }

            for(auto &tmp:idx.out){
                std::cout<<tmp<<" ";
            }
            std::cout<<std::endl;
        }
    }

    return 0;
}
