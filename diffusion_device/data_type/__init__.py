# -*- coding: utf-8 -*-
"""
Folder of modules defining:

data = load_data(metadata, infos)
data = process_data(data, metadata, settings, infos)
profiles = get_profiles(data, metadata, settings, infos)
radius, fits = size_profiles(profiles, metadata, settings, infos)
plot_and_save(radius, profiles, fits, outpath, settings, infos)
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
