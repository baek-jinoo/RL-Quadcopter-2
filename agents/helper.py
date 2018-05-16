import shutil
import os
import time
import ntpath

def mv_file_to_dir_with_date(origin_file_path, directory_path):
    if not os.path.isfile(origin_file_path):
        return

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    filename, extension = ntpath.splitext(ntpath.basename(origin_file_path))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    destination_filename = timestr + '-' + filename + extension
    destination_file_path = os.path.join(directory_path, destination_filename)

    shutil.move(origin_file_path,
                destination_file_path)

