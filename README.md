# ply2atti-scanline
Scripts for extraction fo attitudes from outcrop 3D models.

# Usage

## ply2atti

```
Usage: ply2atti.py -f input_filename [options] [color1 color2 ... colorN] [-o output_filename]

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit
  -f FILE, --file=FILE  input painted 3d model
  -o FILE, --outfile=FILE
                        output color coded 3d model, for use with
                        --colorencode
  -j, --join            joins all resultant data in a single file, instead of
                        a file for each color as default. Recomended if using
                        --eigen option.
  -n, --network         Outputs each different colored plane, through graph
                        analysis.

  Calibration Options:
    These are small utilities to aid calibration of your data.

    -e, --eigen         outputs only the third eigenvector of each color
                        points.
    -a COLOR:AZIMUTH, --azimuth=COLOR:AZIMUTH
                        calibrates your output data by turning its azimuth
                        horizontaly until the given color has the given
                        dipdirection
    -u VALUE, --value=VALUE
                        Determines the value used for the color encode option.
                        Defaults to 0.90.
```

##scanline

```
Virtual scanline analysis system, for use with meshlab point picking tool.
Usage:

python proc_sline.py file_name

Prints to stdout the resultant data.
```