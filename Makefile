PYTHON := .venv/bin/python

.PHONY: lint run video

lint:
	$(PYTHON) -m flake8 src

run:
	$(PYTHON) src/main.py

video:
	ffmpeg -r 30 -f image2 -s 1080x720 -i video/cam1_frame_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p simulation_video_cam1.mp4
	ffmpeg -r 30 -f image2 -s 1080x720 -i video/cam2_frame_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p simulation_video_cam2.mp4
	rm -rf video
