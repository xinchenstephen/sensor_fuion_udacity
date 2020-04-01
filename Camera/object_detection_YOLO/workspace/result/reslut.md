# 1. Look at the coco.cfg file and find out which object classes the YOLO network is able to detect. Then, find some interesting images containing some of those objects and load them into the framework. Share some of the results with us if you like.

the result is in the file. I download a pic from th website and we can see that YOLO works well on the detection of the pic

# 2.Experiment with the size of the blob image and use some other settings instead of 416 x 416. Measure the execution time for varying sizes of the blob image.

the size  the time  detection number
416         1.364          most
732         3.0            less
832         3.55           least

# 3.periment with the confidence threshold and the NMS threshold. How do the detection results change for different settings of both variables?

confidence threshold:

the larger threshold is, the less object will show on the img because those with low confidence result will dispear.

NMS threshold:

the less NMS threshold is, the less object will have the overlapped pic

the results' pic are all in the result file folder.