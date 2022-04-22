# hazen

### Overview
hazen is a software framework for performing automated analysis of magnetic resonance imaging (MRI) Quality Assurance data.

It provides automatic quantitative analysis for the following measurements of MRI phantom data:
- Signal-to-noise ratio (SNR)
- Spatial resolution
- Slice position and width
- Uniformity
- Ghosting
- MR Relaxometry


### Usage

The CLI version of hazen is designed to be pointed at single folders containing DICOM file(s). Example datasets are provided in the `tests/data/` directory. Run the CLI version of hazen with the following example commands.

To perform an SNR measurement on the provided example Philips DICOMs:

`hazen snr tests/data/snr/Philips`

To perform a spatial resolution measurement on example data provided by the East Kent Trust:

`hazen spatial_resolution tests/data/resolution/philips`

To see the full list of available tools, enter:

`hazen -h`

The `--report` option provides additional information for some of the functions. For example, the user can gain additional insight into the performance of the snr function by entering:

`hazen snr tests/data/snr/Philips --report`


### GitHub

The hazen source code can be viewed on the [GSTT-CSC hazen Github page](https://github.com/GSTT-CSC/hazen). If you have any problems or feedback using hazen, please raise an Issue on the Github repo. 
