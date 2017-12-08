import rosbag
bag = rosbag.Bag('backseatdriver_TTIC_2017-12-08-06-28-25.bag')
for topic, msg, t in bag.read_messages(topics=['backseatdriver/tag_detection']):
    print (topic)
bag.close()