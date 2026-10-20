# Driving SDK

<p align="center">
  English | <a href="./README.md">简体中文</a>
</p>

## Latest Updates

* [Mar. 30, 2026]: 🚀 The BEVFusion and Sparse4D models now support mixed precision.
* [Mar. 30, 2026]: 🚀 The SparseConv3D operator now supports fp16, and the MSDA operator supports dim values greater than 64.
* [Mar. 30, 2026]: 🚀 The GraphSoftmax and SparseInverseConv3D operators are now supported.
* [Mar. 30, 2026]: 🚀 Performance has been optimized for the ScatterAdd and SubMSparseConv3D operators.

## Software Introduction

Driving SDK is an operator and model acceleration library based on the Ascend NPU platform for autonomous driving scenarios, vision-language-action (VLA) for embodied AI, and world models. It provides a set of high-performance operators and model migration examples and supports the PyTorch framework. For details, see [Introduction](docs/en/introduction/Introduction.md).

## Directory Structure

```shell
DrivingSDK
├── ci                          # CI scripts
├── cmake                       # CMake scripts
├── docker                      # Custom image building
├── docs                        # Documentation
├── include                     # Header files
├── kernels                     # Operator implementations
│  ├── add_relu
│  │  ├── op_host              # Host-side implementation
│  │  ├── op_kernel            # Device-side implementation
│  ├── ...
│  └── CMakeLists.txt
├── model_examples              # Model examples
│   ├── BEVFormer               # BEVFormer model example
│   ├── BEVFusion               # BEVFusion model example
│   └── ...
├── mx_driving                  # Acceleration library core
│   ├── csrc                    # Acceleration library API adaptation layer
│   ├── dataset                 # Load balancing feature
│   ├── modules                 # Sparse modules
│   ├── ops                     # High-performance APIs
│   ├── patcher                 # One-click patcher
│   └── ...
├── onnx_plugin                 # ONNX framework adaptation layer
├── pre-commit                  # pre-commit configuration
├── scripts                     # Project scripts
├── tests                       # Test files
├── CMakeLists.txt              # CMake configuration file
├── CMakePresets.json           # CMake configuration file
├── LICENSE                     # Open-source license
├── OWNERS                      # Code review
├── README.md                   # Project documentation
├── requirements.txt            # Environment dependencies
├── setup.py                    # Wheel packaging configuration
└── Third_Party_Open_Source_Software_Notice  # Third-party open-source software notice
```

## Version Description

For the CPU architectures supported by Driving SDK operators and the corresponding versions of Python, PyTorch, and TorchNPU, see [Version Notes](docs/en/version/versions.md).

## Installation Guide

The community provides four deployment methods: container deployment, pip installation, custom image building, and source code compilation. You are advised to deploy the Driving SDK environment in a container for a quick start. For the detailed deployment process, see [Deploying the Driving SDK Environment](docs/en/installation/installation.md).

## Quick Start

This guide explains how to use the Driving SDK high-performance APIs and how to perform model migration and optimization with Driving SDK. See the [Quick Start Guide](docs/en/get_started/quick_start_guide.md).

## Feature Highlights

* Through the patching mechanism of Python, you can quickly and cleanly migrate code that was originally built for GPU platforms to Ascend-adapted, NPU-optimized implementations. For details, see [One-click Patcher](docs/en/features/patcher.md).

* Driving SDK provides multiple types of high-performance APIs, including general, sampling, voxelization, detection, sparse, and fusion APIs. For details, see [API](docs/en/api/README.md).

* It supports converting ONNX models to OM models through the ATC conversion tool. For details, see [Convert ONNX to OM](docs/en/features/onnx_example.md).

* Driving SDK implements load balancing for autonomous driving models. Load balancing strategies relieve bottlenecks caused by uneven compute loads and improve model performance. For details, see [Driving SDK Load Balancing Feature](docs/en/features/dataload_balance.md).

## Model Usage Guide

The Driving SDK repository provides migration and optimization examples for autonomous driving models on Ascend servers, including perception, planning, end-to-end, and VLA models. Each model has detailed usage guidance, and the repository will continue to add and refine typical models. For details, see the [Supported Models](docs/en/models/support_list.md).

## Model Migration Guide

Through the one-click patcher, Driving SDK provides one-click migration from GPU models to the Ascend NPU, helping developers quickly adapt their models. For details, see [Model Migration Guide](docs/en/migration_tuning/model_migration.md).

## Model Performance Tuning Guide

Driving SDK provides systematic performance tuning solutions, including framework feature configuration, high-performance operator replacement, and Host Bound issue optimization. For details, see [Model Performance Tuning Guide](docs/en/migration_tuning/model_optimization.md).

## Driving SDK Branch Maintenance Strategy

The maintenance stages for Driving SDK version branches are as follows:

| Status | Duration | Description |
| ------------------- | ------------------- | ------------------------------------------------ |
| Planning | 1 to 3 months | Planned features. |
| Development | 3 months | Develop features. |
| Maintenance | 6 to 12 months | Merge all resolved issues and release versions. Different Driving SDK versions follow different maintenance policies. Regular releases and long-term support releases have maintenance cycles of 6 months and 12 months, respectively. |
| No Maintenance | 0 to 3 months | Merge all resolved issues. No dedicated maintenance staff. No version releases. |
| End of Life (EOL) | N/A | The branch no longer accepts any changes. |

## Driving SDK Version Maintenance Strategy

| Driving SDK Version | Maintenance Strategy | Current Status | Release Date | Subsequent Status | EOL Date |
|---------------------|-----------|---------|------------|--------------------|-----------|
| v26.1.0 | Regular release | Maintenance | June 30, 2026 | Expected to become no-maintenance starting December 30, 2026. | |
| v26.0.0 | Regular release | Maintenance | March 30, 2026 | Expected to become no-maintenance starting September 30, 2026. | |
| v7.3.0 | Regular release | No Maintenance | December 30, 2025 | No maintenance starting June 30, 2026. | |
| v7.2.RC1 | Regular release | No Maintenance | September 30, 2025 | No maintenance starting March 30, 2026. | June 30, 2026 |
| v7.1.RC1 | Regular release | No Maintenance | June 30, 2025 | No maintenance starting December 30, 2025. | March 30, 2026 |
| v7.0.RC1 | Regular release | No Maintenance | March 30, 2025 | No maintenance starting September 30, 2025. | December 30, 2025 |
| v6.0.0 | Regular release | End of Life | December 30, 2024 | No maintenance starting June 30, 2025. | September 30, 2025 |
| v6.0.0-RC3 | Regular release | End of Life | September 30, 2024 | No maintenance starting March 30, 2025. | June 30, 2025 |
| v6.0.0-RC2 | Regular release | End of Life | June 30, 2024 | No maintenance starting December 30, 2024. | March 30, 2025 |
| v6.0.0-RC1 | Regular release | End of Life | March 30, 2024 | No maintenance starting September 30, 2024. | December 30, 2024 |

## FAQ

If you encounter issues while using Driving SDK, you can check the [Driving SDK FAQ](./docs/en/faq/faq.md) to resolve them yourself, or leave a message in [Issues](https://gitcode.com/Ascend/DrivingSDK/issues).

## Contact Us

The future roadmap will be updated dynamically in [Driving SDK Roadmap](https://gitcode.com/Ascend/DrivingSDK/issues/132). You are welcome to use this link to interact with us and share your requests.

To exchange development experience, share usage tips, and receive project updates promptly, we have created the official Driving SDK WeChat group.

Whether you are using this project now or have creative ideas, you are welcome to join. 👋

<p align="center"> <img src="./docs/en/figures/DrivingSDK_wechat_qrcode.jpg" width="150"> </p>

## Security Statement

This section mainly describes the security hardening information, public IP address information, and communication matrix of Driving SDK. For details, see [Driving SDK Security Statement](./SECURITYNOTE_en.md).

## Disclaimer

### For Driving SDK Users

1. The models provided by Driving SDK are for non-commercial use only.
2. For each model, the Driving SDK platform only suggests datasets that you can use for training. Huawei does not provide any datasets. If you use these datasets for training, ensure that you comply with the corresponding dataset license. If infringement disputes arise from your use of the datasets, Huawei bears no responsibility.
3. If you find any issues while using Driving SDK models, including but not limited to functional issues and compliance issues, submit an issue on Gitcode. We will review and resolve it promptly.
4. The models provided by Driving SDK are for reference design only. The third-party open-source software that the models depend on is provided and maintained by the corresponding third-party communities. Issues arising from third-party open-source software usually need to be fixed by the relevant communities. The Driving SDK repository does not promise to fix issues in the third-party open-source software itself, nor does it guarantee that it will test, verify, or fix all vulnerabilities and errors in the third-party open-source software.

### For Dataset Owners

If you do not want your dataset to be mentioned in Driving SDK models, or if you want to update the description of your dataset in a Driving SDK model, submit an issue on Gitcode. We will delete or update your dataset description according to your request. We sincerely thank you for your understanding and contribution to Driving SDK.

## License

* The Driving SDK product usage license. See [LICENSE](./LICENSE).<br>
* The documents in the `docs` directory of the Driving SDK tools are covered by the CC-BY 4.0 license. See [LICENSE](./docs/LICENSE).

## Contribution Statement

Before contributing, please sign the [Open Project Contributor License Agreement (CLA)](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930).
Driving SDK typically releases four official versions each year.
If you encounter a bug, please [submit an issue](https://gitcode.com/Ascend/DrivingSDK/issues).
If you plan to contribute bug fixes, submit a pull request. See the [specific requirements](./CONTRIBUTING_en.md#pull-request).
If you plan to contribute new features or functionality, please create an issue first so that we can discuss it. Please describe the model or operator name, the request background and purpose, the design approach, and the impact on the existing repository. Submitting a PR without prior discussion may cause the request to be rejected because the project direction may differ from your proposal.
For a more detailed contribution process, see [Contribution Guide](./CONTRIBUTING_en.md).

## Acknowledgments

Driving SDK is jointly contributed by the following departments of Huawei:

* Ascend Computing Training Development Dept

Thanks to every PR from the community. Contributions to Driving SDK are welcome!
