/*
 * @Author: your name
 * @Date: 2021-02-04 20:14:07
 * @LastEditTime: 2021-02-04 20:17:21
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /Deploy/common/logger.cpp
 */
/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "logger.h"

#include "logging.h"

namespace Trt {
Logger gLogger{Logger::Severity::kINFO};
LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

void setReportableSeverity(Logger::Severity severity) {
  gLogger.setReportableSeverity(severity);
  gLogVerbose.setReportableSeverity(severity);
  gLogInfo.setReportableSeverity(severity);
  gLogWarning.setReportableSeverity(severity);
  gLogError.setReportableSeverity(severity);
  gLogFatal.setReportableSeverity(severity);
}
}  // namespace Trt
