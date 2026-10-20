#!/bin/bash

# 控制CANN日志是否输出到终端（0:保存到文件，1:打印到终端）
export ASCEND_SLOG_PRINT_TO_STDOUT=0
# 设置CANN全局日志级别（0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR, 4:NULL）
export ASCEND_GLOBAL_LOG_LEVEL=3
# 是否开启Event日志（0:关闭, 1:开启）
export ASCEND_GLOBAL_EVENT_ENABLE=0
# 是否开启HCCL异步错误处理（0:关闭, 1:开启）
export HCCL_ASYNC_ERROR_HANDLING=0
# 设置HCCL算子执行超时时间，单位秒（此处为9000秒）
export HCCL_EXEC_TIMEOUT=9000
# 是否关闭HCCL通信白名单校验（0:开启校验, 1:关闭校验）
export HCCL_WHITELIST_DISABLE=1
# 是否开启FFTS高性能通信调度器（TRUE:开启, FALSE:关闭）
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
# 设置HCCL建立连接的超时时间，单位秒（此处为1200秒）
export HCCL_CONNECT_TIMEOUT=1200
# 是否开启多流内存复用（1:开启, 0:关闭）
export MULTI_STREAM_MEMORY_REUSE=1
# 设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1
# 设置是否开启taskque,0-关闭/1-开启/2-流水优化
export TASK_QUEUE_ENABLE=2
# 设置是否开启combined标志,0-关闭/1-开启
export COMBINED_ENABLE=1
# 启用可扩展内存段分配策略
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#配置初始化模式
export ACL_OP_INIT_MODE=0
#避免torch2.7权重加载报错，设置weights_only=False
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
