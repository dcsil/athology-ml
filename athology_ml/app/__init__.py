import os

# When using the webservice, we don't want to polute the console with tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
