class Logger:
  def __init__(self, path):
    self.file = open(path, 'a')

  def log(self, log):
    self.file.write(log + '\n')
    self.file.flush()

  def log_shapes(self, X_train, X_test):
    self('Train data shape {}\nTest data shape{}'.format(X_train.shape, X_test.shape))

  def log_accuracy(self, accuracy):
    self('Accuracy {}'.format(accuracy))

  def log_paths(self, pos_path, neg_path):
    self('Pos file {}'.format(pos_path))
    self('Neg file {}'.format(neg_path))

  def close(self):
    self.file.close()

  def __del__(self):
    self.file.close()

  def __call__(self, log_msg):
    self.file.write(log_msg + '\n')
    self.file.flush()