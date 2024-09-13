from mlforkidsimages import MLforKidsImageProject

# treat this key like a password and keep it secret!
key = "43c3ee00-71e8-11ef-85dc-8b7707db00922a561630-5b6f-4fba-a758-d5f608d403e8"

# this will train your model and might take a little while
myproject = MLforKidsImageProject(key)
myproject.train_model()

# CHANGE THIS to the image file you want to recognize
demo = myproject.prediction("rabi.jpeg")

label = demo["class_name"]
confidence = demo["confidence"]

# CHANGE THIS to do something different with the result
print ("result: '%s' with %d%% confidence" % (label, confidence))
