rm runs/detect/*/*/*png
rm runs/detect/*/*/*jpg
rm -r runs/detect/*/*/labels
rm `find . -name last.pt`
rm `find . -name *cache`
rm -r `find . -name *val?`