from diffusion_device.keys import settings


# path to data
datapath = 'SampleData/settings.json'
json_infos = {}

# Radii[m] (min, max, number)
json_infos['Radii[m] (min, max, number)'] = [
    500e-12,
    10e-9,
    20]

# Alternative way of specifying range
json_infos['Radii[m] (min, max, step)'] = None

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

# Normalise the profiles?
json_infos['Normalise the profiles?'] = True

# Option to only analyse a slice of the image
json_infos['Slice [m] (center(±), width(+))'] = None

# For bright Images, should the image be flattened?
json_infos['Flatten bright field'] = False

# For movies, what position should be plotted?
json_infos['Stack images to plot'] = None

# Frames range
json_infos['Frames range'] = [
    None,
    None]

# Ignore Error
json_infos['Ignore Error'] = False

settings.generate_json(datapath, json_infos)