# Model Migration

Driving SDK provides the **One-Click Patcher** capability for efficient and convenient model migration, enabling you to train your model on the Ascend NPU by adding just a few lines of code.

The following steps only guide you on how to quickly migrate models with the One-Click Patcher. For more detailed information about the One-Click Patcher, see [One-Click Patcher](../features/patcher.md).

1. Minimal usage

    Add the following code at the **very top** of the training script (usually `train.py`), before all other `import` statements:

    ```python
    # At the top of train.py
    from mx_driving.patcher import default_patcher
    default_patcher.apply()

    # ↓↓↓ The following is an example of the original model code. Use your actual model, and you do not need to change a single character. ↓↓↓
    import mmcv
    import torch
    from my_model import build_model
    ...
    ```
    
    > [!NOTE]
    >
    > **Why place it at the very top?** The Patcher uses the Monkey Patch mechanism to replace the functions and classes of the target modules. If a target module is imported before `apply()` is called, the patch may not take effect correctly.
    >
    > **About the `torch`/`torch_npu` dependency:** The Patcher imports `torch` and `torch_npu` automatically. You do not need to import these modules manually before importing the Patcher.
2. Run the model normally
