# Copyright (c) 2016-2020 AUEB BaLab
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import glob
from shutil import copyfile

old_days = [x.split(os.sep)[-1] for x in glob.glob('../web-results/results/*')]
new_days = [x.split(os.sep)[-1] for x in glob.glob('../web-results/results2019/*')]

old_No = len(old_days)
new_days = new_days[:old_No]

def move_old_results():
    for filename in glob.glob('../web-results/results/*/*/*/*'):
        #import pdb
        old_filename = filename
        filename = filename.replace('../web-results/results', '../web-results2/results')
        segments = filename.split(os.sep)
        day = segments[-4]
        index = old_days.index(day)
        filename = filename.replace('Friday the 13th (film)','Friday the 13th')
        filename = filename.replace(day,new_days[index])
        filename = filename.replace('google', 'old_google')
        filename = filename.replace('bing','old_bing')
        filename = filename.replace('duckduckgo', 'old_duckduckgo')
        segments = filename.split(os.sep)[:-1]
        dirs = os.sep.join(segments)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        copyfile(old_filename, filename)

move_old_results()

for filename in glob.glob('../web-results/results2019/*/*/*/*'):
    old_filename = filename
    filename = filename.replace('../web-results/results2019', '../web-results2/results')
    segments = filename.split(os.sep)[:-1]
    dirs = os.sep.join(segments)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    copyfile(old_filename, filename)

