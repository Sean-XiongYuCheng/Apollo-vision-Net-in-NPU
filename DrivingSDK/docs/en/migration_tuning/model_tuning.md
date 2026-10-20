# Model Performance Tuning

Driving SDK enables model tuning through model streamlining, quantization, and operator fusion. It achieves load balancing by distributing tasks across multiple compute units, and it also supports custom operators to supplement special compute logic. These methods work together to optimize Driving SDK performance.

  - [Model optimization](./model_optimization.md): Use the TorchNPU framework, the high-performance operator library built into the Driving SDK, and other resources to improve model performance.
  - [Load balancing](../features/dataload_balance.md): Driving SDK uses load balancing strategies to alleviate the performance bottleneck caused by unbalanced compute load and improve model performance.
  - [Custom operators](../operator_adaption/from-scratch.md): Operator logic is fully under your control, which helps you debug and locate issues when precision anomalies or performance bottlenecks occur.
