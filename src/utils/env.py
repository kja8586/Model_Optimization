def configure_tensorflow():
    import os
    import warnings
    import logging
    import tensorflow as tf

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['AUTOGRAPH_VERBOSITY'] = '0'
    os.environ['KMP_WARNINGS'] = '0'

    warnings.filterwarnings('ignore')
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    tf.get_logger().setLevel('ERROR')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
