from diffusion_device.keys import metadata


# path to data
datapath = 'SampleData/UVim300ulph.tif'
json_infos = {}

# Location of the data file(s). If None, will take the last path part of
# datapath.
json_infos['Image file name'] = None

# Background file, relative to the image file. None means no background
json_infos['Background file name'] = 'UVbg.tif'

# image file to remove the background coming from the optics (Not chip related)
json_infos['Optics background file name'] = None

# Image exposition time [s]
json_infos['Image exposition time [s]'] = 1

# Background image exposition time [s]
json_infos['Background image exposition time [s]'] = 1

# Optics background image exposition time [s]
json_infos['Optics background image exposition time [s]'] = None

# Height of the channel [m]
json_infos['Wz[m]'] = 53e-6

# Width of the channel [m]
json_infos['Wy[m]'] = 100e-6

# Width of the walls [m] (Only for multiple channel in an image)
json_infos['Wall Width [m]'] = 100e-6

# Number of channels in a file
json_infos['Number of channels in a file'] = 4

# Flow [ulph]. If None, is taken from the image_filename. !!!Check!!!
json_infos['Q[ulph]'] = 300

# Reading position at the middle of the image [m]
json_infos['Read Positions [m]'] = [
    0.0,
    2.556e-3,
    21.953e-3,
    45.1e-3]

# Pixel Size [m]
json_infos['Pixel Size [m]'] = 833e-9

# order of the region of interest [px]
json_infos['Image border[px] (t, d, l, r)'] = [
    None,
    None,
    None,
    None]

# Date [YYYYMMDD]
json_infos['Date [YYYYMMDD]'] = '20100231'

# Analyte informations
json_infos['Analyte informations'] = '15uM BSA'

# Buffer informations
json_infos['Buffer informations'] = '100mM phosphate pH8'

# Device informations
json_infos['Device informations'] = 'Kadi100x50 model 1'

# Data Type
# One of "4pos", "4pos_stack", "12pos", "scans"
json_infos['Type of data'] = '4pos'

# profile position. None means the mean over Z is used.
json_infos['Z position of scans [m]'] = None

# Where is the flow going for each reading position?
# For scans, only use 'u' up or 'd' down
json_infos['Flow direction (u, d, l, r)'] = [
    'u',
    'd',
    'u',
    'd']

# Operator
json_infos['Operator'] = 'Kadi'

# Success [1-3]
json_infos['Success [1-3]'] = 3

# Temperature [K]
json_infos['Temperature [K]'] = 295

# Viscosity [Pa s]
json_infos['Viscosity [Pa s]'] = 1e-3

metadata.generate_json(datapath, json_infos)
