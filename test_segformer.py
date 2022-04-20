import cv2 as cv
import glob
import os.path as osp


expected_path = "output/expected"
actual_path = "output/actual"

expected_files = {osp.splitext(osp.basename(f))[0]: f for f in glob.glob(osp.join(expected_path, "*.png"))}
actual_files = {osp.splitext(osp.basename(f))[0]: f for f in glob.glob(osp.join(actual_path, "*.png"))}

for name, path in actual_files.items():
	assert name in expected_files, "A file was found that is not expected"
	
	actual = cv.imread(path)
	expected = cv.imread(expected_files[name])
	assert actual == expected

print("Correct")	

