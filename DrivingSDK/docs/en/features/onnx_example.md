# Training-Inference Integration Example

## Overview

Using the `MultiScaleDeformableAttn` (MSDA) operator as an example, this document shows the full flow from exporting a model to ONNX to running the OM model. It provides a basic ONNX example for this repository.

## Single-Operator ONNX Export Script

Use a Python script to export the single-operator model. The export script is as follows:

```python
import os
import torch
import torch_npu
import mx_driving
from mx_driving import multi_scale_deformable_attn

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, value, shapes, level_start_index, sampling_locations, attention_weights):
        return multi_scale_deformable_attn(value, shapes, level_start_index, sampling_locations, attention_weights)


def onnx_export(model, inputs, onnx_model_name,
                input_names=None, output_names=None):
    if input_names is None:
        input_names = ["input_names"]
    if output_names is None:
        output_names = ["output_names"]
    model.eval()
    OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX
    with torch.no_grad():
        torch.onnx.export(model, inputs,
                            onnx_model_name,
                            opset_version=11,
                            operator_export_type=OPERATOR_EXPORT_TYPE,
                            input_names=input_names,
                            output_names=output_names)


def export_onnx(name):
    bs, num_levels, num_heads, num_points, num_queries, embed_dims = 2, 1, 8, 4, 40000, 32
    shapes = torch.tensor([[200, 200] * num_levels]).reshape(num_levels, 2).long()

    num_keys = sum((H * W).item() for H, W in shapes)
    value = torch.rand(bs, num_keys, num_heads, embed_dims) * 0.01
    sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2) * 1.2 - 0.1
    attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
    level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1])).long()

    value = value.half()
    sampling_locations = sampling_locations.half()
    attention_weights = attention_weights.half()

    npu_value = value.clone().detach().npu()
    npu_sampling_locations = sampling_locations.clone().detach().npu()
    npu_attention_weights = attention_weights.clone().detach().npu()
    npu_level_start_index = level_start_index.clone().detach().npu()
    npu_shapes = shapes.clone().detach().npu()

    model = Model().npu()
    model(npu_value, npu_shapes, npu_level_start_index, npu_sampling_locations, npu_attention_weights)
    onnx_export(model, (npu_value, npu_shapes, npu_level_start_index, npu_sampling_locations, npu_attention_weights), name,
        ["value", "shapes", "level_start_index", "sampling_locations", "attention_weights"], ["outputs"])

if __name__ =='__main__':
    export_onnx("./msda.onnx")
```

After it runs, the `msda.onnx` file is generated in the current folder.

## Domain Conversion

If the domains in the ONNX model are not unified, install the conversion repository and perform domain conversion. Otherwise, this operation is not required.
The conversion repository is [domain conversion repository](https://gitee.com/Ronnie_zheng/MagicONNX). The conversion script is as follows:

```python
from magiconnx import OnnxGraph
graph = OnnxGraph('msda.onnx')

graph.keep_default_domain()
graph.save('msda.onnx')
```

After it runs, it unifies the multiple domains in the ONNX model.

## Converting an ONNX Model into an OM Model

Use ATC to convert the ONNX model into an OM model. Before you run the command, set the environment variables shown below. You can check `soc_version` with `npu-smi info`:

```shell
pip3 show mx_driving # Record the displayed result as $mx_driving_path
export ASCEND_CUSTOM_OPP_PATH=$mx_driving_path/mx_driving/packages/vendors/customize/
export LD_LIBRARY_PATH=$mx_driving_path/mx_driving/packages/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
atc --framework 5 --output msda --soc_version Ascend910B2 --model msda.onnx --op_select_implmode high_precision --precision_mode must_keep_origin_dtype --log debug
```

If `ATC run success, welcome to the next use.` appears after the command runs, the OM model conversion succeeds.

## Generating Input Data

Build the input files according to the model inputs used during ONNX export. The script is as follows:

```python
import os
import numpy as np
import torch
import torch_npu

bs, num_levels, num_heads, num_points, num_queries, embed_dims = 2, 1, 8, 4, 40000, 32
shapes = torch.tensor([[200, 200] * num_levels]).reshape(num_levels, 2).long()

num_keys = sum((H * W).item() for H, W in shapes)
level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1])).long()

input1 = np.random.rand(bs, num_keys, num_heads, embed_dims).astype(np.float16)
input2 = shapes.numpy().astype(np.int64)
input3 = level_start_index.numpy().astype(np.int64)
input4 = np.random.rand(bs, num_queries, num_heads, num_levels, num_points, 2).astype(np.float16)
input5 = np.random.rand(bs, num_queries, num_heads, num_levels, num_points).astype(np.float16)

# Generate the input .bin files
os.makedirs("./inputs", exist_ok=True)
input1.tofile("./inputs/input1.bin")
input2.tofile("./inputs/input2.bin")
input3.tofile("./inputs/input3.bin")
input4.tofile("./inputs/input4.bin")
input5.tofile("./inputs/input5.bin")
```

After it runs, the individual input files for the model are generated in the `inputs` folder.

## Running the OM Model

Clone the [msame tool](https://gitee.com/ascend/tools/tree/master/msame) repository and install it according to the README file.

Then use the `msame` tool to feed the generated input files into the OM model for execution.

```shell
./msame --model ./msda.om --input ./inputs/input1.bin,./inputs/input2.bin,./inputs/input3.bin,./inputs/input4.bin,./inputs/input5.bin --output ./msame/out/ --outfmt BIN --loop 1
```

After it runs, the OM model inference results are generated in the `./msame/out` folder.

## Accuracy Verification

This section verifies whether the OM model inference results match the single-operator results. You can choose any tool for verification. The example code is as follows. `msda_output_0.bin` in the code is the name of the model inference result generated after the OM model runs.

```python
import numpy as np
import torch, torch_npu
import mx_driving
from mx_driving import multi_scale_deformable_attn

bs, num_levels, num_heads, num_points, num_queries, embed_dims = 2, 1, 8, 4, 40000, 32

input1 = np.fromfile("./inputs/input1.bin", dtype=np.float16).reshape(bs, -1, num_heads, embed_dims)
input2 = np.fromfile("./inputs/input2.bin", dtype=np.int64).reshape(num_levels, 2)
input3 = np.fromfile("./inputs/input3.bin", dtype=np.int64).reshape(num_levels)
input4 = np.fromfile("./inputs/input4.bin", dtype=np.float16).reshape(bs, num_queries, num_heads, num_levels, num_points, 2)
input5 = np.fromfile("./inputs/input5.bin", dtype=np.float16).reshape(bs, num_queries, num_heads, num_levels, num_points)

input1_npu = torch.from_numpy(input1).npu()
input2_npu = torch.from_numpy(input2).npu()
input3_npu = torch.from_numpy(input3).npu()
input4_npu = torch.from_numpy(input4).npu()
input5_npu = torch.from_numpy(input5).npu()

golden = multi_scale_deformable_attn(input1_npu, input2_npu, input3_npu, input4_npu, input5_npu)
output = torch.from_numpy(np.fromfile("msda_output_0.bin", dtype=np.float16).reshape(bs, num_queries, num_heads * embed_dims))

print(golden)
print(output)
```

## FAQ

### `No Parser Is Registered for Op`

Possible cause 1: The `mx_driving` package compiled in Docker may not include the ONNX plugin. You need to compile `protoc`. See [Driving SDK Installation](../installation/installation.md#prerequisites-1) to compile and install it.

Possible cause 2: The `mx_driving` environment variables used during ONNX-to-OM conversion were not imported. Therefore, the corresponding operator was not found during conversion and the process failed.

### `Cannot Find Node xxx Custom infer_datatype Function`

Cause: At present, only some operators in this repository implement `inferShape` and `inferDtype`. Refer to the `IMPL_OP_INFERSHAPE` code lines in the `.cpp` file of the MSDA operator in the `kernels/multi_scale_deformable_attn/op_host/` directory and add similar logic to resolve the issue.

### `Optype xxx of ops kernel is unsupported`

Cause: The given input types do not match the input types supported by the operator. Check the model input types.

### `The model has 2 domain_version fields, but only one is allowed`

Cause: Domain conversion has not been performed. See the example in the preceding section for the conversion process.
