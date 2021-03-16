/*
 * @Author: your name
 * @Date: 2021-03-14 09:54:01
 * @LastEditTime: 2021-03-16 19:53:20
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Inference/src/classify.cpp
 */
#include "../include/classify.h"

bool classify::constructNetwork(
    TrtUniquePtr<nvinfer1::IBuilder>& builder,
    TrtUniquePtr<nvinfer1::INetworkDefinition>& network,
    TrtUniquePtr<nvinfer1::IBuilderConfig>& config,
    TrtUniquePtr<nvonnxparser::IParser>& parser) {
  // parser onnx
  auto parsed = parser->parseFromFile(
      locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
      static_cast<int>(Trt::gLogger.getReportableSeverity()));
  if (!parsed) {
    return false;
  }

  config->setMaxWorkspaceSize(16_MiB);
  if (mParams.fp16) {
    config->setFlag(BuilderFlag::kFP16);
  }
  if (mParams.int8) {
    config->setFlag(BuilderFlag::kINT8);
    TrtCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
  }

  TrtCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

  return true;
}

bool classify::engine_exists(const std::string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

bool classify::build_engine() {
  //   builder
  mBuilder = TrtUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(Trt::gLogger.getTRTLogger()));
  if (!mBuilder) {
    return false;
  }

  // network
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  mNetwork = TrtUniquePtr<nvinfer1::INetworkDefinition>(
      mBuilder->createNetworkV2(explicitBatch));
  if (!mNetwork) {
    return false;
  }

  mConfig =
      TrtUniquePtr<nvinfer1::IBuilderConfig>(mBuilder->createBuilderConfig());
  if (!mConfig) {
    return false;
  }

  mParser = TrtUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*mNetwork, Trt::gLogger.getTRTLogger()));
  if (!mParser) {
    return false;
  }

  // construct
  auto constructed = constructNetwork(mBuilder, mNetwork, mConfig, mParser);
  if (!constructed) {
    return false;
  }

  // engine
  if (!engine_exists(mYamlConfig["engine"].as<std::string>())) {
    Trt::gLogInfo << "Building engine ..." << std::endl;
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mBuilder->buildEngineWithConfig(*mNetwork, *mConfig),
        TrtCommon::InferDeleter());
    if (!mEngine) {
      Trt::gLogInfo << "Building engine failed!" << std::endl;
      return false;
    }
    Trt::gLogInfo << "Saveing engine ..." << std::endl;
    saveEngine(*mEngine, mYamlConfig["engine"].as<std::string>(),
               Trt::gLogInfo);
    Trt::gLogInfo << "Engine saved!" << std::endl;
    Trt::gLogInfo << "Engine loaded!" << std::endl;
  } else {
    Trt::gLogInfo << "Loading engine ..." << std::endl;
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        loadEngine(mYamlConfig["engine"].as<std::string>(), 0, std::cout),
        TrtCommon::InferDeleter());
    Trt::gLogInfo << "Engine loaded!" << std::endl;
  }
  if (!mEngine) {
    return false;
  }

  // context
  mContext = std::shared_ptr<nvinfer1::IExecutionContext>(
      mEngine->createExecutionContext(), TrtCommon::InferDeleter());

  mBuffers = TrtCommon::BufferManager(mEngine);

  assert(mNetwork->getNbInputs() == 1);
  mInputDims = mNetwork->getInput(0)->getDimensions();
  assert(mInputDims.nbDims == 4);

  assert(mNetwork->getNbOutputs() == 1);
  mOutputDims = mNetwork->getOutput(0)->getDimensions();
  assert(mOutputDims.nbDims == 4);

  return true;
}

bool classify::infer(cv::Mat& frame, int& out) {
  // Read the input data into the managed buffers
  Trt::gLogInfo << "Preprocessing ..." << std::endl;
  assert(mParams.inputTensorNames.size() == 1);

  time_point PreprocessingStartTime{std::chrono::high_resolution_clock::now()};
  if (!preprocess(mBuffers, frame)) {
    return false;
  }
  time_point PreprocessingEndTime{std::chrono::high_resolution_clock::now()};
  Trt::gLogInfo
      << "Preprocessing in "
      << duration(PreprocessingEndTime - PreprocessingStartTime).count()
      << " sec." << std::endl;

  mBuffers.copyInputToDeviceAsync();

  Trt::gLogInfo << "Inferenceing ..." << std::endl;
  time_point InferenceingStartTime{std::chrono::high_resolution_clock::now()};
  bool status = mContext->executeV2(mBuffers.getDeviceBindings().data());
  if (!status) {
    return false;
  }
  time_point InferenceingEndTime{std::chrono::high_resolution_clock::now()};
  Trt::gLogInfo << "Inferenceing in "
                << duration(InferenceingEndTime - InferenceingStartTime).count()
                << " sec." << std::endl;

  mBuffers.copyOutputToHostAsync();

  Trt::gLogInfo << "Postprocessing ..." << std::endl;
  if (!postprocess(mBuffers, frame, out)) {
    return false;
  }

  return true;
}

bool classify::preprocess(const TrtCommon::BufferManager& buffers,
                          cv::Mat& frame) {
  const int channels = mInputDims.d[1];
  const int inputH = mInputDims.d[2];
  const int inputW = mInputDims.d[3];

  if (frame.empty()) {
    Trt::gLogInfo << "Input image load failed\n";
    return false;
  }

  auto input_size = cv::Size(inputW, inputH);
  cv::Mat resizeFrame;
  cv::Mat rgbFrame;
  cv::resize(frame, resizeFrame, input_size, 0, 0, cv::INTER_NEAREST);
  cv::cvtColor(resizeFrame, rgbFrame, cv::COLOR_BGR2RGB);
  cv::Mat normalizeFrame;
  rgbFrame.convertTo(normalizeFrame, CV_32FC3, 1.f / 255.f);

  std::vector<cv::Mat> frame_channels;
  cv::split(normalizeFrame, frame_channels);
  frame_channels[0] = (frame_channels[0] - 0.485) / 0.229;
  frame_channels[1] = (frame_channels[1] - 0.456) / 0.224;
  frame_channels[2] = (frame_channels[2] - 0.406) / 0.225;
  cv::merge(frame_channels, normalizeFrame);

  float* hostDataBuffer =
      static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));

  std::vector<cv::Mat> chw;
  for (size_t i = 0; i < channels; ++i) {
    chw.emplace_back(
        cv::Mat(input_size, CV_32FC1, hostDataBuffer + i * inputW * inputH));
  }
  cv::split(normalizeFrame, chw);

  return true;
}

bool classify::postprocess(const TrtCommon::BufferManager& buffers,
                           cv::Mat& frame, int& out) {
  const int outputSize = 1 * 2;
  float* output =
      static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
  std::vector<float> result{output, output + outputSize};

  if (result[0] > result[1]) {
    out = 0;
  }
  if (result[0] < result[1]) {
    out = 1;
  }

  return true;
}
