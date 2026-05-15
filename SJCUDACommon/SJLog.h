#pragma once
#include "SJCUDACommon.h"

#define LOGGING(level, format, ...) SJLog::PrintLOG(level, __FILE__, __func__, __LINE__, format, __VA_ARGS__)
enum class LOG_LEVEL {
	NOTHING = 0,
	WARNING = 1,
	INFO = 2,
	DETAIL = 3,
	VERBOSE = 4
};
class SJ_CUDA_COMMON_API SJLog
{

public:
	static LOG_LEVEL m_level;

	SJLog();
	~SJLog();
	static void SetLOGLevel(const LOG_LEVEL level);
	static void PrintLOG(const LOG_LEVEL level, const char* file, const char* func, const int line_num, const char* format, ...);
};
