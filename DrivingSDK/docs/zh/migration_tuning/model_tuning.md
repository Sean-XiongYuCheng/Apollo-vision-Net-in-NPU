# 模型性能调优

Driving SDK可通过模型精简量化与算子融合开展模型调优，依托多算力单元任务分发实现负载均衡，同时还支持自定义算子补齐特殊计算逻辑，多种方式协同实现Driving SDK的性能优化。

  - [模型优化](./model_optimization.md)：可借助TorchNPU框架、Driving SDK内置的高性能算子库等提升模型性能。
  - [负载均衡](../features/dataload_balance.md)：Driving SDK通过负载均衡策略，缓解计算负载不均衡瓶颈，提升模型性能。
  - [自定义算子](../operator_adaption/from-scratch.md)：算子逻辑完全自主可控，在精度异常、性能瓶颈时有助于调试定位。
