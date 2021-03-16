/*
 * @Author: your name
 * @Date: 2021-03-14 09:53:52
 * @LastEditTime: 2021-03-14 13:45:12
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Inference/include/classify.h
 */
#ifndef _CLASSIFY_H_
#define _CLASSIFY_H_

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "../common/TrtEngines.h"
#include "../common/TrtUtils.h"
#include "../common/argsParser.h"
#include "../common/buffers.h"
#include "../common/common.h"
#include "../common/logger.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "yaml-cpp/yaml.h"

using namespace nvinfer1;
using namespace Trt;

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
using duration = std::chrono::duration<float>;

class classify {
 private:
  TrtCommon::OnnxTrtParams mParams;
  YAML::Node mYamlConfig;

  nvinfer1::Dims mInputDims;
  nvinfer1::Dims mOutputDims;

  TrtUniquePtr<nvinfer1::IBuilder> mBuilder;
  TrtUniquePtr<nvinfer1::INetworkDefinition> mNetwork;
  TrtUniquePtr<nvonnxparser::IParser> mParser;
  TrtUniquePtr<nvinfer1::IBuilderConfig> mConfig;

  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
  std::shared_ptr<nvinfer1::IExecutionContext> mContext;

  TrtCommon::BufferManager mBuffers;

 private:
  bool constructNetwork(TrtUniquePtr<nvinfer1::IBuilder>& builder,
                        TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
                        TrtUniquePtr<nvinfer1::IBuilderConfig>& config,
                        TrtUniquePtr<nvonnxparser::IParser>& parser);

  bool engine_exists(const std::string& name);
  bool preprocess(const TrtCommon::BufferManager& buffers, cv::Mat& frame);
  bool postprocess(const TrtCommon::BufferManager& buffers, cv::Mat& frame,
                   int& out);

 public:
  classify(const TrtCommon::OnnxTrtParams& params, YAML::Node& yamlConfig)
      : mParams(params), mEngine(nullptr), mYamlConfig(yamlConfig) {}
  ~classify(){};
  bool build_engine();
  bool infer(cv::Mat& frame, int& out);
};

#endif