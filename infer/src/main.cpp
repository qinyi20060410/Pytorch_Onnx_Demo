/*
 * @Author: your name
 * @Date: 2021-03-14 09:49:07
 * @LastEditTime: 2021-03-16 19:55:58
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Inference/src/main.cpp
 */

#include "../include/classify.h"

const std::string gTrtName = "TensorRT.vpsIferFromOnnx";

TrtCommon::OnnxTrtParams classify_initializeTrtParams(YAML::Node& config) {
  TrtCommon::OnnxTrtParams params;
  params.dataDirs.push_back(config["onnx_path"].as<std::string>());
  params.onnxFileName = config["onnx_name"].as<std::string>();
  params.inputTensorNames.push_back(config["input_name"].as<std::string>());
  params.outputTensorNames.push_back(config["output_name"].as<std::string>());
  params.dlaCore = std::stoi(config["useDLACore"].as<std::string>());
  params.int8 = std::stoi(config["runInInt8"].as<std::string>());
  params.fp16 = std::stoi(config["runInFp16"].as<std::string>());
  return params;
}

int main(int argc, char** argv) {
  auto TrtTest = Trt::gLogger.defineTest(gTrtName, argc, argv);
  Trt::gLogger.reportTestStart(TrtTest);

  YAML::Node classify_yamlConfig =
      YAML::LoadFile("../config/classify_config.yaml");
  if (!classify_yamlConfig) {
    Trt::gLogError << "Can't read configure file!" << std::endl;
    return Trt::gLogger.reportFail(TrtTest);
  }

  classify Trt_classify(classify_initializeTrtParams(classify_yamlConfig),
                        classify_yamlConfig);
  if (!Trt_classify.build_engine()) {
    return Trt::gLogger.reportFail(TrtTest);
  }

  return Trt::gLogger.reportPass(TrtTest);
}