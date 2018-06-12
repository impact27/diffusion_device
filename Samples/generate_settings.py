from diffusion_device.keys import settings


# path to data
datapath = 'SampleData/settings.json'
json_infos = {}

# Radii[m] (min, max, number)
json_infos['Radii[m] (min, max, number)'] = [
    500e-12,
    10e-9,
    20]

# Radii in log
json_infos['Radii in log'] = False

# Number of species to fit
json_infos['Number of species to fit'] = 1

# Ignore Edge[m]
json_infos['Ignore Edge[m]'] = 0

# Positions to fit. None means all. Useful to ignore one of the channels
# In a multiple channel image
json_infos['Pos to fit'] = None

# Number of z slices
json_infos['Number of z slices'] = 11

# Method to process first profile. Leave 'none' for now.
json_infos['First Position Filter'] = 'none'

# Option to only analyse a slice of the image
json_infos['Slice [m] (center(±), width(+))'] = None

# For bright Images, should the image be flattened?
json_infos['Flatten bright field'] = True

# For movies, what position should be plotted?
json_infos['Stack images to plot'] = None

# Frames range
json_infos['Frames range'] = [
    None,
    None]

# Ignore Error
json_infos['Ignore Error'] = False

# Apply a Savitzky-Golay filter. None = no filter
json_infos['Savitzky-Golay filter (window, order)'] = None

# For optimisation and identical processing
json_infos['Stack is stationnary'] = False

# If there is corners or random blobs, True
json_infos['Background had good features'] = True

# If LSE/ signal over noise > 1, ignore
json_infos['Remove bad fits'] = False

# Set to 0<x<1 to reduce fitting error from step size.
json_infos['dx factor'] = 1

# Can get a better fit if the offset is not the same for all profiles.
json_infos['Vary offset?'] = False

# For a single scan, the slice to consider
json_infos['Scan Slice'] = [
    None,
    None]

# If too many profiles, rebin
json_infos['Rebin Profiles'] = 1

# Should the alignment be done with the background or the image?
json_infos['Use Image Coordinates'] = True

# Decrease the number of images by rebinning
json_infos['Rebin stack'] = 1

settings.generate_json(datapath, json_infos)
