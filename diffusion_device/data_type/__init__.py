# -*- coding: utf-8 -*-
"""
Folder of modules defining:

data, state = load_data(metadata)
data, pixel_size, *infos = process_data(data, metadata, settings)
profiles = get_profiles(metadata, settings, data, pixel_size, *infos)
radius, fits, error = size_profiles(profiles, pixel_size, metadata, settings)
plot_and_save(radius, profiles, fits, error, pixel_size, outpath, settings)
savedata(data, outpath)

Copyright (C) 2017  Quentin Peter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from . import multi_channels_image
