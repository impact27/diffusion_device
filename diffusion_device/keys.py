# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:37:30 2017

@author: quentinpeter
"""

from .json_info_list import ListGenerator

metadata = ListGenerator(
    'metadata', "SampleData/UVim300ulph.tif", True, "Image file name")

metadata.add_info("KEY_MD_FN", "Image file name", "path",
                  required=True,
                  example=None,
                  explanation="Location of the data file(s). If None, will take the last path part of datapath.")
metadata.add_info("KEY_MD_BGFN", "Background file name", "path",
                  required=False,
                  explanation="Background file, relative to the image file. None means no background",
                  example="UVbg.tif")
metadata.add_info("KEY_MD_OPBGFN", "Optics background file name", "path",
                  required=False,
                  explanation="image file to remove the background coming from the optics (Not chip related)")
metadata.add_info("KEY_MD_EXP", "Image exposition time [s]", float,
                  required=False,
                  example=1)
metadata.add_info("KEY_MD_BGEXP", "Background image exposition time [s]", float,
                  required=False,
                  example=1)
metadata.add_info("KEY_MD_OPBGEXP", "Optics background image exposition time [s]",
                  float,
                  required=False)
metadata.add_info("KEY_MD_WZ", "Wz[m]", float,
                  required=True,
                  explanation="Height of the channel [m]",
                  example=53e-6)
metadata.add_info("KEY_MD_WY", "Wy[m]", float,
                  required=True,
                  explanation="Width of the channel [m]",
                  example=100e-6)
metadata.add_info("KEY_MD_WALLWIDTH", "Wall Width [m]", float,
                  required=False,
                  explanation="Width of the walls [m] (Only for multiple channel in an image)",
                  example=100e-6)
metadata.add_info("KEY_MD_NCHANNELS", "Number of channels in a file", int,
                  required=False,
                  example=4)
metadata.add_info("KEY_MD_Q", "Q[ulph]", float,
                  required=True,
                  regexp='([\d\.]+)ul?L?O?p?_?-?h',
                  explanation="Flow [ulph]. If None, is taken from the image_filename. !!!Check!!!",
                  example=300)
metadata.add_info("KEY_MD_RPOS", "Read Positions [m]", float,
                  required=True,
                  explanation="Reading position at the middle of the image [m]",
                  example=[
                      0.0,
                      2.556e-3,
                      21.953e-3,
                      45.1e-3
                  ])
metadata.add_info("KEY_MD_PIXSIZE", "Pixel Size [m]", float,
                  required=True,
                  example=833e-9)
metadata.add_info("KEY_MD_BORDER", "Image border[px] (t, d, l, r)", int,
                  required=False,
                  default=[None, None, None, None],
                  explanation="order of the region of interest [px]")
metadata.add_info("KEY_MD_DATE", "Date [YYYYMMDD]", str,
                  required=False,
                  example="20100231")
metadata.add_info("KEY_MD_ANALYTE", "Analyte informations", str,
                  required=False,
                  example="15uM BSA")
metadata.add_info("KEY_MD_BUFFER", "Buffer informations", str,
                  required=False,
                  example="100mM phosphate pH8")
metadata.add_info("KEY_MD_DEVICE", "Device informations", str,
                  required=False,
                  example="Kadi100x50 model 1")
metadata.add_info("KEY_MD_TYPE", "Type of data", str,
                  required=True,
                  example="4pos",
                  explanation="Data Type\n"
                  '# One of "4pos", "4pos_stack", "12pos", "scans"')
metadata.add_info("KEY_MD_SCANZ", "Z position of scans [m]", float,
                  required=False,
                  explanation="profile position. None means the mean over Z is used.")
metadata.add_info("KEY_MD_FLOWDIR", "Flow direction (u, d, l, r)", str,
                  required=True,
                  example=['u', 'd', 'u', 'd'],
                  explanation="Where is the flow going for each reading position?\n"
                  "# For scans, only use 'u' up or 'd' down")
metadata.add_info("KEY_MD_OPERATOR", "Operator", str,
                  required=False,
                  example="Kadi")
metadata.add_info("KEY_MD_SUCCESS", "Success [1-3]", int,
                  required=False,
                  example=3)
metadata.add_info("KEY_MD_T", "Temperature [K]", float,
                  required=False,
                  default=295)
metadata.add_info("KEY_MD_ETA", "Viscosity [Pa s]", float,
                  required=False,
                  default=1e-3)

settings = ListGenerator('settings', 'SampleData/settings.json', False)
settings.add_info("KEY_STG_R", "Radii[m] (min, max, number)", float,
                  required=False,
                  example=[
                      5e-10,  # Min [m]
                      1e-08,  # Max [m]
                      20  # number
                  ])
settings.add_info("KEY_STG_R_STEP", "Radii[m] (min, max, step)", float,
                  required=False,
                  legacy=True,
                  explanation="Alternative way of specifying range")
settings.add_info("KEY_STG_RLOG", "Radii in log", bool,
                  required=False,
                  default=False)
settings.add_info("KEY_STG_NSPECIES", "Number of species to fit", int,
                  required=False,
                  example=1)
settings.add_info("KEY_STG_IGNORE", "Ignore Edge[m]", float,
                  required=False,
                  default=0)
settings.add_info("KEY_STG_FITPOS", "Pos to fit", int,
                  required=False,
                  explanation=("Positions to fit. None means all."
                               " Useful to ignore one of the channels\n"
                               "# In a multiple channel image"))
settings.add_info("KEY_STG_ZGRID", "Number of z slices", int,
                  required=False,
                  default=11
                  )
settings.add_info("KEY_STG_POS0FILTER", "First Position Filter", str,
                  required=False,
                  default='none',
                  explanation="Method to process first profile. Leave 'none' for now.")
settings.add_info("KEY_STG_NORMALISE", "Normalise the profiles?", bool,
                  required=False,
                  default=True,
                  )
settings.add_info("KEY_STG_SLICE", "Slice [m] (center(±), width(+))", float,
                  required=False,
                  explanation="Option to only analyse a slice of the image")
settings.add_info("KEY_STG_BRIGHT_FLAT", "Flatten bright field", bool,
                  required=False,
                  default=True,
                  explanation="For bright Images, should the image be flattened?")
settings.add_info("KEY_STG_STACK_POSPLOT", "Stack images to plot", int,
                  required=False,
                  explanation="For movies, what position should be plotted?")
settings.add_info("KEY_STG_STACK_FRAMESSLICES", "Frames range", int,
                  required=False,
                  default=[None, None])
settings.add_info("KEY_STG_IGNORE_ERROR", "Ignore Error", bool,
                  required=False,
                  default=False)
settings.add_info("KEY_STG_SGFILTER", "Savitzky-Golay filter (window, order)", int,
                  required=False,
                  default=None,
                  explanation="Apply a Savitzky-Golay filter. None = no filter")
settings.add_info("KEY_STG_STAT_STACK", "Stack is stationnary", bool,
                  required=False,
                  default=False,
                  explanation="For optimisation and identical processing")
settings.add_info("KEY_STG_GOODFEATURES", "Background had good features", bool,
                  required=False,
                  default=True,
                  explanation="If there is corners or random blobs, True ")
settings.add_info("KEY_STG_SUPERFLATTEN", "Flatten using high degree polynomial", bool,
                  required=False,
                  default=False,
                  explanation="If the result is definitely not flat")


