# Release Notes

## Version Mapping

### Product Version

<table><tbody><tr><th class="firstcol" valign="top" width="26.25%"><p>Product</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p><span>Driving SDK</span></p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Version</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers=><p>26.1.0</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Version Type</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>Official release</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Release Date</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>July 2026</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Maintenance</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>6 months</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
> For information about Driving SDK version maintenance, see the [Branch Maintenance Policy](../../../README_en.md#driving-sdk-branch-maintenance-strategy).

### Related Product Versions

**Table 1**  Driving SDK software version compatibility table

|Driving SDK Branch|CANN Version|TorchNPU Version|Python Version|PyTorch Version|
|--|--|--|--|--|
|master(under development)|in development|in development|3.9, 3.10, 3.11|2.7.1|
|branch_v26.1.0|9.1.0|26.1.0|3.9, 3.10, 3.11|2.7.1|
|branch_v26.0.0|9.0.0|26.0.0|3.9, 3.10, 3.11|2.7.1|

> [!NOTE]
> You can choose a Driving SDK code branch as needed, then download the source code and install it.

## Version Compatibility

> [!NOTE]
>
> In the tables in this section, "/" indicates that the versions cannot be used together, and "Y" indicates that they can.

**Table 2**  Driving SDK and TorchNPU version compatibility

<table style="table-layout: fixed; width: 750px">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">Driving SDK</th>
      <th colspan="4">TorchNPU Version</th>
    </tr>
    <tr>
      <th>7.2.0</th>
      <th>7.3.0</th>
      <th>26.0.0</th>
      <th>26.1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.1.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

**Table 3**  Driving SDK and CANN version compatibility

<table style="table-layout: fixed; width: 750px">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">Driving SDK</th>
      <th colspan="4">CANN Version</th>
    </tr>
    <tr>
      <th>8.3.RCX</th>
      <th>8.5.X</th>
      <th>9.0.X</th>
      <th>9.1.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.0.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.1.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

## Important Notes

None

## Update Notes

### New Features

- All Driving SDK operators support <term>Ascend 950 products</term>.
- Autonomous driving models BEVFormer, BEVFusion, Sparse4D, MapTRv2, UniAD, and SparseDrive support <term>Ascend 950 products</term>.
- Added the GR00T-N1.6 model.
- Added the LingBot-VLA model.
- Added the StarVLA model.
- VLA models GR00T-N1.6 and Pi0.5 support <term>Ascend 950 products</term>.

### Deleted Features

None

### Interface Change Notes

None

### Resolved Issues

None

### Known Issues

None

## Upgrade Impact

### Impact on the System During the Upgrade

- Impact on services:

    The software version upgrade interrupts services.

- Impact on network communication:

    There is no impact on network communication.

### Impact on the System After the Upgrade

None

## Documentation

|Document|Description|Update Notes|
|--|--|--|
|[Software Installation](../installation/installation.md)|Guides you through installing Driving SDK on the NPU, covering hardware and OS compatibility, software version compatibility, driver and firmware installation, and three installation methods: image installation, online installation, and source code installation. It helps you quickly set up the Driving SDK development environment.|-|
|[Quick Start](../get_started/quick_start_guide.md)|Uses the `scatter_max` operator and the BEVFusion model as examples to guide developers who are new to Driving SDK through high-performance operator invocation and model training tasks on the NPU, helping them quickly get started with distributed training of autonomous driving models.|-|

## Virus Scan Results and Vulnerability Patch List

### Virus Scan Results

| Antivirus Software | Antivirus Software Version | Virus Database Version | Scan Time | Scan Result |
|---|---|---|---|---|
| QiAnXin | 8.0.5.5260 | 2026-07-05 08:00:00.0 | 2026-07-06 | No viruses or malware |
| Kaspersky | 12.0.0.6672 | 2026-07-06 10:03:00 | 2026-07-06 | No viruses or malware |
| Bitdefender | 7.5.1.200224 | 7.101158 | 2026-07-06 | No viruses or malware |

### Fixed Vulnerabilities

None
